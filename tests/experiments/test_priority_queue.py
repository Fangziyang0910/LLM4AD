import json
from experiments.infra.priority_queue import dependencies_finished


def test_only_all_finished_dependencies_release_queue(tmp_path):
    a, b = tmp_path/'a.json', tmp_path/'b.json'
    assert not dependencies_finished([a,b])
    a.write_text(json.dumps({'plan':[{'status':'finished'}]}))
    for state in ('queued','running','paused','blocked','stopped'):
        b.write_text(json.dumps({'plan':[{'status':state}]}))
        assert not dependencies_finished([a,b])
    b.write_text(json.dumps({'plan':[]}))
    assert not dependencies_finished([a,b])
    b.write_text(json.dumps({'plan':[{'status':'finished'}]}))
    assert dependencies_finished([a,b])
