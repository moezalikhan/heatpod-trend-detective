"""One module per investigation step. Each has render(case)."""

from dataclasses import dataclass

from config.catalog import Variable


@dataclass(frozen=True)
class Case:
    region: str          # region key, e.g. "florida"
    region_info: dict    # data/results/<region>/region.json
    var: Variable
    summary: dict        # data/results/<region>/<code>/summary.json

    @property
    def code(self) -> str:
        return self.var.code
