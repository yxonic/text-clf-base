from text_clf_base.data import TextClfDataset
from text_clf_base.util import get_tokenizer


def test_dataset():
    max_length = 64
    tokenizer = get_tokenizer()
    tokenizer.enable_padding(length=max_length)
    tokenizer.enable_truncation(max_length=max_length)
    dataset = TextClfDataset(
        "data/sample/train_text.txt",
        "data/sample/train_label.txt",
        tokenizer,
    )

    assert len(dataset) == 5372
    assert len(dataset.data) == len(dataset.label)
    data, label = dataset[0]
    assert len(data) == max_length
    assert label == 0
