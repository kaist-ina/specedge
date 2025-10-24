import os

import torch

import util


class _ConfigMeta(type):
    _initialized = False

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    def __getattr__(cls, name):
        if name.startswith("_"):
            return super().__getattribute__(name)

        if not cls._initialized:
            cls._initialize()

        if name in cls.__dict__:
            return cls.__dict__[name]
        else:
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    def __setattr__(cls, name, value):
        super().__setattr__(name, value)

    def _initialize(cls):
        raise NotImplementedError("Subclasses must implement _initialize()")

    def _from_env(cls, key: str):
        value = os.getenv(key)

        if value is None or value == "null":
            raise ValueError(f"Environment variable '{key}' is not set")

        return value

    def reset(cls):
        cls._initialized = False
        cls._initialize()


class SpecEdgeClientConfig(metaclass=_ConfigMeta):
    """
    Configuration for the SpecEdge client

    Results and logs are stored in the directory
    "result_path/exp_name/process_name/seed"

    Attributes:
        optimization (int): Optimization level for the model
        result_path (str): Path to the directory where the results will be stored
        exp_name (str): Name of the experiment
        process_name (str): Name of the process

        seed (int): Seed for the random number generator

        draft_model (str): Path to the draft model
        device (torch.device): Device to run the model on
        dtype (torch.dtype): Data type to use for the model

        dataset (str): Name of the dataset

        max_n_beams (int): Maximum number of beams to generate
        max_beam_len (int): Maximum length of a beam
        max_branch_width (int): Maximum width of a branch
        max_budget (int): Maximum budget for the SpecExec algorithm

        proactive_type (str): Type of proactive draft
        proactive_max_n_beams (int): Maximum number of beams to generate proactively
        proactive_max_beam_len (int): Maximum length of a beam for proactive draft
        proactive_max_branch_width (int): Maximum width of a branch for proactive draft
        proactive_max_budget (int): Maximum budget for the proactive draft

        max_new_tokens (int): Maximum number of new tokens to generate
        max_request_num (int): Maximum number of requests to send

        host (str): Hostname of the server
        req_idx (int): Index of the request
    """

    @classmethod
    def _initialize(cls):
        # experiment configuration
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.optimization = int(cls._from_env("SPECEDGE_OPTIMIZATION"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))

        # model configuration
        cls.draft_model = cls._from_env("SPECEDGE_DRAFT_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.reasoning = cls._from_env("SPECEDGE_REASONING") == "True"

        # dataset configuration
        cls.dataset = cls._from_env("SPECEDGE_DATASET")

        # SpecExec configuration
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))
        cls.max_beam_len = int(cls._from_env("SPECEDGE_MAX_BEAM_LEN"))
        cls.max_branch_width = int(cls._from_env("SPECEDGE_MAX_BRANCH_WIDTH"))
        cls.max_budget = int(cls._from_env("SPECEDGE_MAX_BUDGET"))

        # proactive draft configuration
        cls.proactive_type = cls._from_env("SPECEDGE_PROACTIVE_TYPE")
        cls.proactive_max_n_beams = int(cls._from_env("SPECEDGE_PROACTIVE_MAX_N_BEAMS"))
        cls.proactive_max_beam_len = int(
            cls._from_env("SPECEDGE_PROACTIVE_MAX_BEAM_LEN")
        )
        cls.proactive_max_branch_width = int(
            cls._from_env("SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH")
        )
        cls.proactive_max_budget = int(cls._from_env("SPECEDGE_PROACTIVE_MAX_BUDGET"))

        # token generation configuration
        cls.max_new_tokens = int(cls._from_env("SPECEDGE_MAX_NEW_TOKENS"))
        cls.max_request_num = int(cls._from_env("SPECEDGE_MAX_REQUEST_NUM"))
        cls.req_offset = int(cls._from_env("SPECEDGE_REQ_OFFSET"))
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))

        # server configuration
        cls.host = cls._from_env("SPECEDGE_HOST")
        cls.client_idx = int(cls._from_env("SPECEDGE_CLIENT_IDX"))

        cls._initialized = True


class SpecEdgeBatchClientConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        # experiment configuration
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))

        # model configuration
        cls.draft_model = cls._from_env("SPECEDGE_DRAFT_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_CLIENT_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))

        # dataset configuration
        cls.dataset = cls._from_env("SPECEDGE_DATASET")

        # SpecExec configuration
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))
        cls.max_beam_len = int(cls._from_env("SPECEDGE_MAX_BEAM_LEN"))
        cls.max_branch_width = int(cls._from_env("SPECEDGE_MAX_BRANCH_WIDTH"))
        cls.max_budget = int(cls._from_env("SPECEDGE_MAX_BUDGET"))

        # token generation configuration
        cls.max_batch_size = int(cls._from_env("SPECEDGE_MAX_BATCH_SIZE"))
        cls.max_new_tokens = int(cls._from_env("SPECEDGE_MAX_NEW_TOKENS"))
        cls.max_request_num = int(cls._from_env("SPECEDGE_MAX_REQUEST_NUM"))

        # server configuration
        cls.host = cls._from_env("SPECEDGE_HOST")
        cls.req_idx = int(cls._from_env("SPECEDGE_REQ_IDX"))
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))

        cls._initialized = True


class SpecEdgeServerConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.optimization = int(cls._from_env("SPECEDGE_OPTIMIZATION"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))

        # model configuration
        cls.target_model = cls._from_env("SPECEDGE_TARGET_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.temperature = float(cls._from_env("SPECEDGE_TEMPERATURE"))

        # engine configuration
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))

        cls._initialized = True


class SpecEdgeBatchServerConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))
        cls.batch_type = cls._from_env("SPECEDGE_BATCH_TYPE")
        cls.dataset = cls._from_env("SPECEDGE_DATASET")
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))
        cls.req_offset = int(cls._from_env("SPECEDGE_REQ_OFFSET"))

        # model configuration
        cls.target_model = cls._from_env("SPECEDGE_TARGET_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_SERVER_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.temperature = float(cls._from_env("SPECEDGE_TEMPERATURE"))

        # engine configuration
        cls.max_batch_size = int(cls._from_env("SPECEDGE_MAX_BATCH_SIZE"))
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))
        cls.max_budget = int(cls._from_env("SPECEDGE_MAX_BUDGET"))
        cls.num_clients = int(cls._from_env("SPECEDGE_NUM_CLIENTS"))
        cls.cache_prefill = cls._from_env("SPECEDGE_CACHE_PREFILL") == "True"

        cls._initialized = True


class AutoregressiveBatchConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))

        cls.model = cls._from_env("SPECEDGE_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.temperature = float(cls._from_env("SPECEDGE_TEMPERATURE"))

        cls.dataset = cls._from_env("SPECEDGE_DATASET")

        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))
        cls.max_new_tokens = int(cls._from_env("SPECEDGE_MAX_NEW_TOKENS"))
        cls.max_request_num = int(cls._from_env("SPECEDGE_MAX_REQUEST_NUM"))
        cls.batch_size = int(cls._from_env("SPECEDGE_BATCH_SIZE"))
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))

        cls._initialized = True
