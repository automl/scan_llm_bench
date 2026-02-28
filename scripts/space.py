import neps

class SpacePipeline(neps.PipelineSpace):
    d_model = neps.Integer(
        lower=1,
        upper=16,
    )

    n_layers = neps.Integer(
        lower=1,
        upper=24,
    )

    n_heads = neps.Integer(
        lower=1,
        upper=8,
    )

    n_tokens = neps.Integer(
        lower=50_000_000,  # 50M tokens
        upper=5_000_000_000,  # 5B tokens
        log=True,
    )

    @classmethod
    def with_overrides(cls, **overrides):
        class CustomPipeline(cls):
            pass

        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name != 'with_overrides':
                attr_value = getattr(cls, attr_name)
                if not callable(attr_value):
                    setattr(CustomPipeline, attr_name, attr_value)

        for key, value in overrides.items():
            setattr(CustomPipeline, key, value)

        return CustomPipeline()
