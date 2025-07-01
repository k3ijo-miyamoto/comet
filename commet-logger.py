class CometLogger:
    def __init__(
        self,
        project=None,
        workspace=None,
        api_key=None,
        name=None,
        save_dir=None,
        config=None,
        val_dataset=None,
        num_eval_images=100,
        log_checkpoints=False,
        **kwargs
    ):
        try:
            import comet_ml
            self.comet_ml = comet_ml
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install comet_ml using: pip install comet_ml")

        self.project = project
        self.workspace = workspace
        self.api_key = api_key
        self.name = name
        self.save_dir = save_dir
        self.config = config
        self.val_dataset = val_dataset
        self.num_log_images = min(num_eval_images, len(val_dataset)) if val_dataset else 0
        self.log_checkpoints = log_checkpoints in ("True", "true")
        self.kwargs = kwargs

        self._experiment = None  # lazy init
        self._comet_init = dict(
            api_key=self.api_key,
            project_name=self.project,
            workspace=self.workspace,
            auto_param_logging=False,
            auto_metric_logging=False,
        )
        self._comet_init.update(**kwargs)

        # lazy init 呼び出し（configやname登録）
        _ = self.experiment
        if self.name:
            self.experiment.set_name(self.name)
        if self.config:
            self.experiment.log_parameters(self.config)

    @property
    def experiment(self):
        if self._experiment is None:
            try:
                self._experiment = self.comet_ml.get_global_experiment()
                if self._experiment is None:
                    raise ValueError
            except Exception:
                self._experiment = self.comet_ml.Experiment(**self._comet_init)
        return self._experiment

    def log_metrics(self, metrics, step=None):
        import torch
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.experiment.log_metric(k, v, step=step)

    def save_checkpoint(self, save_dir, model_name, is_best, metadata=None):
        import os
        if not self.log_checkpoints:
            return
        filename = os.path.join(save_dir, model_name + "_ckpt.pth")
        aliases = ["latest"]
        if is_best:
            aliases.append("best")
        if metadata and "epoch" in metadata:
            aliases.append(f"epoch-{metadata['epoch']}")
        self.experiment.log_model(
            name=f"{model_name}_ckpt",
            file_or_folder=filename,
            metadata=metadata or {}
        )

    def finish(self):
        self.experiment.end()

    @classmethod
    def initialize_comet_logger(cls, args, exp, val_dataset):
        comet_params = dict()
        prefix = "comet-"
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            if k.startswith(prefix):
                try:
                    comet_params.update({k[len(prefix):]: int(v)})
                except ValueError:
                    comet_params.update({k[len(prefix):]: v})
        return cls(config=vars(exp), val_dataset=val_dataset, **comet_params)
