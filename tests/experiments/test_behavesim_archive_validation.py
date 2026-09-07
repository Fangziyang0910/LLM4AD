from experiments.behavesim_archive_validation.run import uniform_time_sample


def test_archive_sample_uses_time_only_and_includes_endpoints():
    nodes = [
        {"id": index, "evaluation_id": index * 3, "fitness": (-1) ** index * index}
        for index in range(20)
    ]
    selected = uniform_time_sample(list(reversed(nodes)), 6)

    assert len(selected) == 6
    assert selected[0]["id"] == 0
    assert selected[-1]["id"] == 19
    assert [node["evaluation_id"] for node in selected] == sorted(
        node["evaluation_id"] for node in selected
    )
