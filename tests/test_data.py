from text_clf_base.data import TextClfDataset


def test_dataset():
    dataset = TextClfDataset("data/sample/train_text.txt", "data/sample/train_label.txt")
    assert len(dataset) == 5472
    assert len(dataset.data) == len(dataset.label)
    data, label = dataset[0]
    assert len(data) == 128
    assert label == 0
