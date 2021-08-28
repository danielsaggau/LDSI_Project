reconstructed_model = keras.models.load_model("hist")
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)
