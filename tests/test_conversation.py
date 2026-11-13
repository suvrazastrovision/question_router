from conversation import recent_history


def test_context_copies_messages_and_strips_display_metadata():
    original = [{"role": "user", "content": "What is the hippocampus?", "category": "HIPPOCAMPUS"},
                {"role": "assistant", "content": "A brain region involved in memory."}]
    context = recent_history(original)
    assert context == [{"role": "user", "content": "What is the hippocampus?"},
                       {"role": "assistant", "content": "A brain region involved in memory."}]
    context[0]["content"] = "Changed copy"
    assert original[0]["content"] == "What is the hippocampus?"
    assert recent_history() == []
