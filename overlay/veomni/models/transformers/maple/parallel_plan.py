from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan(qat_ternary: bool = True):
    suffix = ".parametrizations.{}.original" if qat_ternary else ".{}"
    ep_plan = {
        f"model.layers.*.mlp.experts{suffix.format('gate_up_proj')}": Shard(0),
        f"model.layers.*.mlp.experts{suffix.format('down_proj')}": Shard(0),
    }
    plan = ParallelPlan(extra_parallel_plan={"ep": ep_plan})
    plan.extra_parallel_fsdp_no_shard_module = {"ep": {"model.layers.*.mlp.experts"}}
    return plan
