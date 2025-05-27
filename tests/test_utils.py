def test_clean_text():
    from utils.helpers import clean_text
    assert clean_text(" Hello\n") == "Hello"
