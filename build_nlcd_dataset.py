from pathlib import Path
from argparse import ArgumentParser
import json
from build_object_dataset import BuildObjectDataset
from pyproj import CRS
import rasterio
from pyproj import Transformer
import utils
from tqdm import tqdm


all_coarse_labels = {
    11: 0,
    12: 0,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    31: 2,
    41: 3,
    42: 3,
    43: 3,
    51: 4,
    52: 4,
    71: 5,
    72: 5,
    73: 5,
    74: 5,
    81: 6,
    82: 6,
    90: 7,
    95: 7,
}


class BuildNlcdDataset:
    """
    Create nlcd dataset by using the coarse labels from nlcd data website
    Created a map, all_coarse_labels, to convert their labels into continuous labels
    Need to download the land cover data from https://www.mrlc.gov/data/nlcd-2019-land-cover-conus
    Converted crs wkt projection string into lat long using pyproj Transformer library
    """

    def __init__(
        self,
        num_items=None,
        zoom="18",
        nlcd_root: Path = None,
        cvusa_root: Path = None,
        out_root="out/",
    ):
        self.nlcd_root = nlcd_root
        self.cvusa_root = cvusa_root
        self.out_root = out_root
        self.num_items = num_items
        self.zoom = zoom

    def build(self):

        img_path = self.nlcd_root / "nlcd_2019_land_cover_l48_20210604.img"
        if not img_path.is_file():
            raise FileNotFoundError(f"nlcd land cover .img file not found at {path}")
        dataset = rasterio.open(img_path)
        img_file_paths = utils.get_img_file_names(self.cvusa_root)
        num_items = utils.get_num_items(self.num_items, len(img_file_paths))

        nlcd_csv_out_path = self.out_root / f"cvusa_nlcd_{self.num_items}.csv"
        rows = []
        for path in tqdm(img_file_paths[:num_items]):
            lat, lon = utils.get_lat_long_from_fname(path)
            aerial_path = utils.get_aerial_img_from_ground(path, self.zoom)
            labels = self.get_nlcd_labels(dataset, lat, lon)
            try:
                coarse_labels = self.get_nlcd_coarse_labels(labels)
            except KeyError:
                continue
            row = [aerial_path, lat, lon, labels, coarse_labels]
            rows.append(row)

        headers = [
            "aerial_path",
            "latitude",
            "longitude",
            "nlcd_labels",
            "nlcd_coarse_labels",
        ]
        utils.write_csv(headers, rows, nlcd_csv_out_path)

    def get_nlcd_coarse_labels(self, labels):
        coarselabels = []
        for label in labels:
            cl = all_coarse_labels[label]
            if cl in coarselabels:
                continue
            coarselabels.append(cl)
        return coarselabels

    def get_nlcd_labels(self, dataset, lat, lon):
        src_crs = "EPSG:4326"
        t = Transformer.from_crs(src_crs, dataset.crs, always_xy=True)
        x, y = t.transform(lon, lat)
        values = list(rasterio.sample.sample_gen(dataset, [[x, y]]))
        return values[0]

    def get_lat_long(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"json file not found at location {path}")
        with open(path) as f:
            data = json.load(f)
        crs = CRS.from_wkt(data["projection"])
        params = crs.coordinate_operation.params
        lat = params[0].value
        long = params[1].value
        return {"lat": lat, "long": long}


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-ni", "--num_items", type=int, default=None, help="Data size")
    parser.add_argument("-z", "--zoom", type=str, default="18", help="Zoom level")
    parser.add_argument(
        "-nr",
        "--nlcd_root",
        type=str,
        default="/localdisk1/data/nlcd_2019_land_cover",
        help="Nlcd Dataset Root Path",
    )
    parser.add_argument(
        "-cr",
        "--cvusa_root",
        type=str,
        default="/localdisk1/data/cvusa_eag",
        help="CSV Dataset Root Path",
    )
    parser.add_argument(
        "-or",
        "--out_root",
        type=str,
        default="out",
        help="Output Root Dir",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = vars(get_args())
    bnd = BuildNlcdDataset(**args)
    bnd.build()
