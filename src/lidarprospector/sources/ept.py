"""USGS 3DEP airborne lidar (public EPT point clouds on S3): stream only the window, grid the ground
returns. The project name selects the survey, e.g. LA_NortheastDOTD_2_2017."""

from pyproj import Transformer

from ..ept import WEB_MERCATOR, EptProject
from ..terrain import grid_ground
from .base import Dem, utm_epsg


class UsgsEpt:
    licence = "USGS 3DEP public data (see usgs.gov data policy)"
    attribution = "Lidar point cloud data courtesy of the U.S. Geological Survey 3D Elevation Program"
    native_cell = 1.0  # gridded here at 1 m; point density decides what is really resolvable

    def __init__(self, project: str):
        self.project = EptProject(project)
        self.name = f"USGS 3DEP {project}"
        b = self.project.root
        self._ll = Transformer.from_crs(WEB_MERCATOR, 4326, always_xy=True).transform_bounds(
            b[0], b[1], b[3], b[4]
        )
        self.epsg = utm_epsg((self._ll[0] + self._ll[2]) / 2, (self._ll[1] + self._ll[3]) / 2)

    def covers(self, lon: float, lat: float) -> bool:
        return self._ll[0] <= lon <= self._ll[2] and self._ll[1] <= lat <= self._ll[3]

    def fetch(self, bbox, cell: float | None = None) -> Dem:
        if cell not in (None, 1.0):
            raise ValueError(f"{self.name} is gridded at 1 m; cell={cell} is not supported")
        xmin, ymin, xmax, ymax = bbox
        to_merc = Transformer.from_crs(self.epsg, WEB_MERCATOR, always_xy=True)
        corners = [to_merc.transform(x, y) for x in (xmin, xmax) for y in (ymin, ymax)]
        xs, ys = zip(*corners, strict=True)
        win = (min(xs) - 5, min(ys) - 5, max(xs) + 5, max(ys) + 5)
        nodes = self.project.nodes(win)
        if not nodes:
            raise LookupError(f"{self.name} has no points in {bbox}")
        x, y, z, cls = self.project.fetch(nodes, win, self.epsg)
        dtm = grid_ground(x, y, z, cls, bbox, cell=1.0)
        return Dem(dtm, xmin, ymax, 1.0, self.epsg, self.name, self.licence, self.attribution, 1.0)
