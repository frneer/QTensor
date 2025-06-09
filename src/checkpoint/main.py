import tensorflow as tf

from checkpoint.stage import Stage


def dummy_model_function(model, *args, **kwargs):
    """A dummy function to simulate model creation."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(32,)),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    print("Model created with args:", args, "and kwargs:", kwargs)
    return model


if __name__ == "__main__":
    # Example usage of Stage class
    stage = Stage(
        id=0,
        name="Initial Model Creation",
        previous_stage=None,
        function=dummy_model_function,
        function_kwargs={"arg1": "value1", "arg2": "value2"},
    )

    stage.run()
