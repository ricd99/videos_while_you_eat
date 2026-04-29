import pandas as pd
import numpy as np

def _setup_monkeypatch(monkeypatch):
    # Prepare a dummy nn model with kneighbors
    class DummyNN:
        def kneighbors(self, X):
            # Return distances and indices as plain Python lists to avoid numpy dep in tests
            return [[0.1, 0.2, 0.3, 0.4, 0.5]], [[0, 1, 2, 3, 4]]

    # Dummy lookup dataframe to map to channel names/ids
    df_lookup = pd.DataFrame([
        {"channel_name": "ChA", "channel_id": "idA"},
        {"channel_name": "ChB", "channel_id": "idB"},
        {"channel_name": "ChC", "channel_id": "idC"},
        {"channel_name": "ChD", "channel_id": "idD"},
        {"channel_name": "ChE", "channel_id": "idE"},
    ])

    # Patch the loader to return our dummy objects
    monkeypatch.setattr('src.serving.model_repository.get_nn_model', lambda: DummyNN())
    monkeypatch.setattr('src.serving.model_repository.get_lookup_df', lambda: df_lookup)

    # Patch batch_encode used in inference
    monkeypatch.setattr('src.serving.inference.batch_encode', lambda texts: np.array([[0.0, 0.0, 0.0]]))

    # Patch _get_channel_data in the inference module to return a simple input
    monkeypatch.setattr('src.serving.inference._get_channel_data', lambda x: {
        'channel_id': 'C1',
        'channel_name': x,
        'description': 'desc',
        'topics': ['t1'],
        'keywords': 'kw',
        'videos': [{'title': 'video1'}],
        'text': 'desc topics kw video1'
    })

    return df_lookup


def test_inference_with_mock_loader(monkeypatch):
    _setup_monkeypatch(monkeypatch)
    from src.serving import inference as inf

    result = inf.predict("TestChannel")
    assert isinstance(result, list)
    assert len(result) >= 1
    first = result[0]
    assert 'channel_name' in first
    assert 'channel_id' in first
    assert 'similarity_score' in first
