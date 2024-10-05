import random
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch.utils.data import DataLoader, Dataset

from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.diffuse_noise import gen_desired_spatial_coherence, gen_diffuse_noise
from data_loaders.utils.mix import (
    cal_coeff_for_adjusting_relative_energy,
    convolve,
    overlap2,
    pad_or_cut,
    sample_an_overlap,
    sample_ovlp_ratio_and_cal_length,
)
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from data_loaders.utils.window import reverberation_time_shortening_window


def get_spk_file_dict(librispeech_dir: Path) -> Tuple[Dict[str, List[Path]], List[str]]:
    spk_ids = list(map(lambda x: x.name, librispeech_dir.iterdir()))

    spk_file_dict = dict()
    for spk in spk_ids:
        spk_dir = librispeech_dir / spk
        files = list(spk_dir.rglob("*.flac")) + list(spk_dir.rglob("*.wav"))
        spk_file_dict[spk] = files

    return spk_file_dict, spk_ids


def choose_spk_files(
    spk_file_dict: Dict[str, List[Path]],
    spk_ids: List[str],
    num_spk: int,
    data_count: int,
    rng: np.random.Generator = None,
    seed=666,  # only used while rng is None
):
    if rng is None:
        rng = np.random.default_rng(np.random.PCG64(seed))

    spk_files = []
    for _ in range(data_count):
        spks = rng.choice(spk_ids, num_spk)
        spk_files += [list(map(lambda x: rng.choice(spk_file_dict[x]), spks))]
    return spk_files


class LibrispeechDataset(Dataset):

    def __init__(
        self,
        librispeech_dir: str,
        rir_dir: str,
        target: str,
        dataset: str,
        ovlp: str,
        data_count: int,
        speech_overlap_ratio: Tuple[float, float] = [0.1, 1.0],
        sir: Tuple[float, float] = [-5, 5],
        snr: Tuple[float, float] = [10, 20],
        audio_time_len: Optional[float] = None,
        sample_rate: int = 16000,
        num_spk: int = 2,
        noise_type: List[Literal["babble", "white"]] = ["babble", "white"],
        return_noise: bool = False,
        return_rvbt: bool = False,
    ) -> None:
        """The Librispeech dataset

        Args:
            librispeech_dir: a dir contains [wsj_8k_zeromean, sms_wsj.json, ...]
            target:  revb_image, direct_path
            dataset: train_si284, cv_dev93, test_eval92
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert target in ["revb_image", "direct_path"] or target.startswith("RTS"), target
        assert dataset in ["train", "valid", "test"], dataset
        assert ovlp in ["mid", "headtail", "startend", "full", "hms", "fhms"], ovlp
        assert num_spk == 2, ("Not implemented for spk num=", num_spk)
        assert len(set(noise_type) - set(["babble", "white"])) == 0, noise_type

        if ovlp == "full" and audio_time_len == None:
            rank_zero_warn(
                f"dataset {dataset} could not achieve full-overlap without giving a length, the overlap type will be one of startend/headtail/mid-overlap"
            )
            ovlp = "hms"

        self.librispeech_dir = Path(librispeech_dir, dataset).expanduser()
        self.target = target
        self.dataset = dataset
        self.ovlp = ovlp
        self.data_count = data_count
        self.speech_overlap_ratio = speech_overlap_ratio
        self.sir = sir
        self.audio_time_len = audio_time_len
        self.sample_rate = sample_rate
        self.num_spk = num_spk
        self.noise_type = noise_type
        assert sample_rate == 16000, ("Not implemented for sample rate ", sample_rate)

        self.spk_file_dict, self.spk_ids = get_spk_file_dict(self.librispeech_dir)
        self.clean_files = choose_spk_files(self.spk_file_dict, self.spk_ids, self.num_spk, self.data_count)

        self.return_rvbt = return_rvbt
        self.return_noise = return_noise
        self.snr = snr

        self.rir_dir = Path(rir_dir).expanduser() / {"train": "train", "valid": "validation", "test": "test"}[dataset]
        self.rirs = [str(r) for r in list(Path(self.rir_dir).expanduser().rglob("*.npz"))]
        self.rirs.sort()
        # load & save diffuse parameters
        diffuse_paras_path = (Path(rir_dir) / "diffuse.npz").expanduser()
        if diffuse_paras_path.exists():
            self.Cs = np.load(diffuse_paras_path, allow_pickle=True)["Cs"]
        else:
            pos_mics = np.load(self.rirs[0], allow_pickle=True)["pos_rcv"]
            _, self.Cs = gen_desired_spatial_coherence(
                pos_mics=pos_mics, fs=self.sample_rate, noise_field="spherical", c=343, nfft=512
            )
            try:
                np.savez(diffuse_paras_path, Cs=self.Cs)
            except:
                ...
        assert len(self.rirs) > 0, f"{str(self.rir_dir)} is empty or not exists"
        self.shuffle_rir = True if dataset == "train" else False

    def __getitem__(self, index_seed: Tuple[int, int]):
        # for each item, an index and seed are given. The seed is used to reproduce this dataset on any machines
        index, seed = index_seed

        rng = np.random.default_rng(np.random.PCG64(seed))
        num_spk = self.num_spk
        clean_pair = self.clean_files[index]

        # step 1: load single channel clean speech signals
        cleans = []
        for i in range(self.num_spk):
            source, _ = librosa.load(clean_pair[i], sr=self.sample_rate)
            cleans.append(source)
            # assert sr_src == self.sample_rate, (sr_src, self.sample_rate)

        # step 2: load rirs
        if self.shuffle_rir:
            rir_this = self.rirs[rng.integers(low=0, high=len(self.rirs))]
        else:
            rir_this = self.rirs[index % len(self.rirs)]
        rir_dict = np.load(rir_this)
        sr_rir = rir_dict["fs"]
        assert sr_rir == self.sample_rate, (sr_rir, self.sample_rate)

        rir = rir_dict["rir"]  # shape [nsrc,nmic,time]
        assert rir.shape[0] >= num_spk, (rir.shape, num_spk)
        spk_rir_idxs = rng.choice(rir.shape[0], size=num_spk, replace=False).tolist()
        assert len(set(spk_rir_idxs)) == num_spk, spk_rir_idxs
        rir = rir[spk_rir_idxs, :, :]
        if self.target == "direct_path":  # read simulated direct-path rir
            rir_target = rir_dict["rir_dp"]  # shape [nsrc,nmic,time]
            rir_target = rir_target[spk_rir_idxs, :, :]
        elif self.target == "revb_image":  # revb_image
            rir_target = rir  # shape [nsrc,nmic,time]
        elif self.target.startswith("RTS"):  # e.g. RTS_0.1s
            rts_time = float(self.target.replace("RTS_", "").replace("s", ""))
            win = reverberation_time_shortening_window(
                rir=rir, original_T60=rir_dict["RT60"], target_T60=rts_time, sr=self.sample_rate
            )
            rir_target = win * rir
        else:
            raise NotImplementedError("Unknown target: " + self.target)
        num_mic = rir.shape[1]

        # step 3: decide the overlap type, overlap ratio, and the needed length of the two signals
        # randomly sample one ovlp_type if self.ovlp==fhms or hms
        ovlp_type = sample_an_overlap(rng=rng, ovlp_type=self.ovlp, num_spk=num_spk)
        # randomly sample the overlap ratio if necessary and decide the needed length of signals
        lens = [clean.shape[0] for clean in cleans]  # clean speech length of each speaker
        ovlp_ratio, lens, mix_frames = sample_ovlp_ratio_and_cal_length(
            rng=rng,
            ovlp_type=ovlp_type,
            ratio_range=self.speech_overlap_ratio,
            target_len=None if self.audio_time_len is None else int(self.audio_time_len * self.sample_rate),
            lens=lens,
        )

        # step 4: repeat signals if they are shorter than the length needed, then cut them to needed
        cleans = pad_or_cut(wavs=cleans, lens=lens, rng=rng)

        # step 5: convolve rir and clean speech, then place them at right place to satisfy the given overlap types
        rvbts, targets = zip(
            *[
                convolve(wav=wav, rir=rir_spk, rir_target=rir_spk_t, ref_channel=0, align=True)
                for (wav, rir_spk, rir_spk_t) in zip(cleans, rir, rir_target)
            ]
        )
        rvbts, targets = overlap2(rvbts=rvbts, targets=targets, ovlp_type=ovlp_type, mix_frames=mix_frames, rng=rng)

        # step 6: rescale rvbts and targets
        sir_this = None
        if self.sir != None and num_spk == 2:
            sir_this = rng.uniform(low=self.sir[0], high=self.sir[1])  # randomly sample in the given range
            assert len(cleans) == 2, len(cleans)
            coeff = cal_coeff_for_adjusting_relative_energy(wav1=rvbts[0], wav2=rvbts[1], target_dB=sir_this)
            assert coeff is not None
            # scale cleans[1] to -5 ~ 5 dB
            rvbts[1][:] *= coeff
            if targets is not rvbts:
                targets[1][:] *= coeff

        # step 7: generate diffused noise and mix with a sampled SNR
        noise_type = self.noise_type[rng.integers(low=0, high=len(self.noise_type))]
        mix = np.sum(rvbts, axis=0)
        if noise_type == "babble":
            noises = []
            for i in range(num_mic):
                noise_i = np.zeros(shape=(mix_frames,), dtype=mix.dtype)
                noise_list = choose_spk_files(self.spk_file_dict, self.spk_ids, 1, 10, rng=rng)
                for noise_paths in noise_list:
                    noise_ij, _ = librosa.load(noise_paths[0], sr=self.sample_rate)  # [T]
                    # assert sr_noise == self.sample_rate and noise_ij.ndim == 1, (sr_noise, self.sample_rate)
                    noise_i += pad_or_cut([noise_ij], lens=[mix_frames], rng=rng)[0]
                noises.append(noise_i)
            noise = np.stack(noises, axis=0).reshape(-1)
        elif noise_type == "white":
            noise = rng.normal(size=mix.shape[0] * mix.shape[1])
        noise = gen_diffuse_noise(
            noise=noise, L=mix_frames, Cs=self.Cs, nfft=512, rng=rng
        )  # shape [num_mic, mix_frames]

        snr_this = rng.uniform(low=self.snr[0], high=self.snr[1])
        coeff = cal_coeff_for_adjusting_relative_energy(wav1=mix, wav2=noise, target_dB=snr_this)
        assert coeff is not None
        noise[:, :] *= coeff
        snr_real = 10 * np.log10(np.sum(mix**2) / np.sum(noise**2))
        assert np.isclose(snr_this, snr_real, atol=0.5), (snr_this, snr_real)
        mix[:, :] = mix + noise

        # scale mix and targets to [-0.9, 0.9]
        scale_value = 0.9 / max(np.max(np.abs(mix)), np.max(np.abs(targets)))
        mix[:, :] *= scale_value
        targets[:, :] *= scale_value

        paras = {
            "index": index,
            "seed": seed,
            "target": self.target,
            "sample_rate": 16000,
            "dataset": f"Libri_AISHELL/{self.dataset}",
            "noise_type": noise_type,
            "noise": noises if self.return_noise else None,
            "rvbt": rvbts if self.return_rvbt else None,
            "sir": float(sir_this),
            "snr": float(snr_real),
            "ovlp_type": ovlp_type,
            "ovlp_ratio": float(ovlp_ratio),
            "audio_time_len": self.audio_time_len,
            "num_spk": num_spk,
            "rir": {
                "RT60": rir_dict["RT60"],
                "pos_src": rir_dict["pos_src"],
                "pos_rcv": rir_dict["pos_rcv"],
            },
        }

        return torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.float32), paras

    def __len__(self):
        return self.data_count


class LibrispeechDataModule(LightningDataModule):

    def __init__(
        self,
        librispeech_dir: str = "~/datasets/sms_wsj",  # a dir contains [early, noise, observation, rirs, speech_source, tail, wsj_8k_zeromean]
        rir_dir: str = "~/datasets/SMS_WSJ_Plus_rirs",  # containing train, validation, and test subdirs
        target: str = "direct_path",  # e.g. rvbt_image, direct_path
        datasets: Tuple[str, str, str, str] = [
            "train_si284",
            "cv_dev93",
            "test_eval92",
            "test_eval92",
        ],  # datasets for train/val/test/predict
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = [
            4.0,
            4.0,
            None,
            None,
        ],  # audio_time_len (seconds) for train/val/test/predictS
        ovlp: Union[
            str, Tuple[str, str, str, str]
        ] = "mid",  # speech overlapping type for train/val/test/predict: 'mid', 'headtail', 'startend', 'full', 'hms', 'fhms'
        data_counts: Tuple[int, int, int] = [
            40000,
            5000,
            3000,
        ],  # data count for train, valid, test dataset respectively
        speech_overlap_ratio: Tuple[float, float] = [0.1, 1.0],
        sir: Optional[Tuple[float, float]] = [
            -5,
            5,
        ],  # relative energy of speakers (dB), i.e. signal-to-interference ratio
        snr: Tuple[float, float] = [0, 20],  # SNR dB
        num_spk: int = 2,  # separation task: 2 speakers; enhancement task: 1 speaker
        noise_type: List[Literal["babble", "white"]] = ["babble", "white"],  # the type of noise
        return_noise: bool = False,
        return_rvbt: bool = False,
        batch_size: List[int] = [1, 1],  # batch size for [train, val, {test, predict}]
        num_workers: int = 10,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int, int] = [None, 2, 3, 3],  # random seeds for train/val/test/predict sets
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = True,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.librispeech_dir = librispeech_dir
        self.rir_dir = rir_dir
        self.target = target
        self.datasets = datasets
        self.audio_time_len = audio_time_len
        self.ovlp = [ovlp] * 4 if isinstance(ovlp, str) else ovlp
        self.data_counts = data_counts
        self.speech_overlap_ratio = speech_overlap_ratio
        self.sir = sir
        self.snr = snr
        self.num_spk = num_spk
        self.noise_type = noise_type
        self.return_noise = return_noise
        self.return_rvbt = return_rvbt
        self.persistent_workers = persistent_workers

        self.batch_size = batch_size
        while len(self.batch_size) < 4:
            self.batch_size.append(1)

        rank_zero_info("dataset: Libri_AISHELL")
        rank_zero_info(f"train/val/test/predict: {self.datasets}")
        rank_zero_info(f"batch size: train/val/test/predict = {self.batch_size}")
        rank_zero_info(f"audio_time_length: train/val/test/predict = {self.audio_time_len}")
        rank_zero_info(f"target: {self.target}")
        # assert self.batch_size_val == 1, "batch size for validation should be 1 as the audios have different length"
        # assert audio_time_len[2] == None, "the length for test set should be None if you want to test ASR performance"

        self.num_workers = num_workers

        self.collate_func = [collate_func_train, collate_func_val, collate_func_test, default_collate_func]

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, dataset, ovlp, data_count, audio_time_len, seed, shuffle, batch_size, collate_fn):
        ds = LibrispeechDataset(
            librispeech_dir=self.librispeech_dir,
            rir_dir=self.rir_dir,
            target=self.target,
            dataset=dataset,
            ovlp=ovlp,
            data_count=data_count,
            speech_overlap_ratio=self.speech_overlap_ratio,
            sir=self.sir,
            snr=self.snr,
            audio_time_len=audio_time_len,
            num_spk=self.num_spk,
            noise_type=self.noise_type,
            return_noise=self.return_noise,
            return_rvbt=self.return_rvbt,
        )

        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),  #
            batch_size=batch_size,  #
            collate_fn=collate_fn,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[0],
            ovlp=self.ovlp[0],
            data_count=self.data_counts[0],
            audio_time_len=self.audio_time_len[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func[0],
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[1],
            ovlp=self.ovlp[1],
            data_count=self.data_counts[1],
            audio_time_len=self.audio_time_len[1],
            seed=self.seeds[1],
            shuffle=False,
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func[1],
        )

    def test_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[2],
            ovlp=self.ovlp[2],
            data_count=self.data_counts[2],
            audio_time_len=self.audio_time_len[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[3],
            ovlp=self.ovlp[3],
            data_count=self.data_counts[3],
            audio_time_len=self.audio_time_len[3],
            seed=self.seeds[3],
            shuffle=False,
            batch_size=self.batch_size[3],
            collate_fn=self.collate_func[3],
        )


if __name__ == "__main__":
    """python -m data_loaders.librispeech"""
    import importlib

    import torchaudio
    import yaml
    from tqdm import tqdm

    def initialize_module(path: str, args: dict = None, initialize: bool = True):
        idx = path.rfind(".")
        module_path = path[:idx]
        class_or_function_name = path[idx + 1 :]

        module = importlib.import_module(module_path)
        class_or_function = getattr(module, class_or_function_name)

        if initialize:
            if args:
                return class_or_function(**args)
            else:
                return class_or_function()
        else:
            return class_or_function

    dataset = "train"
    # dataset = "valid"
    # dataset = "test"
    save_count, bypass = 100, bool(1)
    save_dir = Path("/home/featurize/data/audio_test/test_mixture")
    with open(r"configs/datasets/librispeech.yaml") as fp:
        config = yaml.safe_load(fp)

    init_args = config["data"]["init_args"]
    init_args["num_workers"] = 1  # for debuging
    datamodule: LightningDataModule = initialize_module(config["data"]["class_path"], init_args)
    datamodule.setup()

    if dataset.startswith("train"):
        dataloader = datamodule.train_dataloader()
    elif dataset.startswith("valid"):
        dataloader = datamodule.val_dataloader()
    elif dataset.startswith("test"):
        dataloader = datamodule.test_dataloader()
    else:
        assert dataset.startswith("predict"), dataset
        dataloader = datamodule.predict_dataloader()

    if type(dataloader) != dict:
        dataloaders = {dataset: dataloader}
    else:
        dataloaders = dataloader

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    for ds, dataloader in dataloaders.items():
        for idx, (noisy, tar, paras) in tqdm(enumerate(dataloader, 1), total=save_count):
            if not bypass:
                torchaudio.save(save_dir / f"{idx:03d}_mix.wav", noisy[0], paras[0]["sample_rate"])
                torchaudio.save(save_dir / f"{idx:03d}_tar.wav", tar[0, :, 0], paras[0]["sample_rate"])

            if idx == save_count:
                break
            ...
