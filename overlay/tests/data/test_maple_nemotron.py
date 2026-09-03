from scripts.maple.prepare_nemotron import valid_messages


def test_nemotron_messages_allow_empty_system_but_not_empty_prompt():
    response = {"role": "assistant", "content": "answer"}

    assert valid_messages([{"role": "system", "content": ""}, {"role": "user", "content": "prompt"}, response])
    assert not valid_messages([{"role": "user", "content": ""}, response])
