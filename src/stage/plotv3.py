#!/usr/bin/env python3

import fnmatch
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def load_pipeline(pipeline_metadata_file, results_path, metadata_path):
    df = pd.DataFrame()
    stages_results = []
    stages_names = []
    with open(pipeline_metadata_file, "r") as f:
        pipeline_metadata = json.load(f)
        for stage_hash in pipeline_metadata["history"]:
            with open(results_path / f"{stage_hash}.json", "r") as stage_file:
                stage_results = json.load(stage_file)
                stages_results.append(stage_results)
            with open(
                metadata_path / f"{stage_hash}.json", "r"
            ) as metadata_file:
                stage_metadata = json.load(metadata_file)
                stages_names.append(stage_metadata["name"])
    df = pd.DataFrame(stages_results)
    df["stage"] = stages_names
    return df


def load_file(hash_, results_path: Path, metadata_path: Path) -> dict:
    """Lee los dos JSON correspondientes a `hash_` y devuelve un dict unificado
    con todas las claves de ambos."""
    # Cargar resultados
    with open(results_path / f"{hash_}.json", "r") as f_res:
        res = json.load(f_res)
    # Cargar metadata
    with open(metadata_path / f"{hash_}.json", "r") as f_meta:
        meta = json.load(f_meta)
    # Combinar, dándole preferencia a `res` en caso de colisión de claves
    combined = {**meta, **res}
    # Añadimos el hash como columna
    combined["hash"] = hash_
    return combined


def flatten_json(obj, parent_key: str = "", sep: str = "_"):
    """Recursively flattens dicts and lists into a single dict mapping
    flattened_key -> value.

    List items get their index injected into the key.
    """
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_json(v, new_key, sep=sep))
    else:
        # reached a leaf
        items[parent_key] = obj
    return items


def add_horizontal_se_band(
    ax,
    mean,
    se,
    *,
    color="red",
    label="Original Model",
    alpha_line=0.8,
    alpha_band=0.2,
    z_mean=3,
    z_line=2,
    z_band=1,
):
    """On ax, draw a dotted line at `mean`, dashed lines at ±se, and fill
    between them."""
    # mean line
    ax.axhline(
        mean,
        color=color,
        linestyle=":",
        linewidth=0.3,
        alpha=alpha_line,
        zorder=z_mean,
        # label=f"{label}"
    )
    # ±1 SE lines
    ax.axhline(
        mean + se,
        color=color,
        linestyle="-",
        linewidth=0.3,
        alpha=0.6,
        zorder=z_line,
    )
    ax.axhline(
        mean - se,
        color=color,
        linestyle="-",
        linewidth=0.3,
        alpha=0.6,
        zorder=z_line,
    )
    # filled band
    x0, x1 = ax.get_xlim()
    ax.fill_between(
        [x0, x1],
        [mean - se, mean - se],
        [mean + se, mean + se],
        color=color,
        alpha=alpha_band,
        zorder=z_band,
    )


if __name__ == "__main__":
    checkpoint_path = Path("checkpoints")
    # pipeline_path = checkpoint_path / "pipelines"
    metadata_path = checkpoint_path / "metadata"
    results_path = checkpoint_path / "results"

    # Patrón: sólo caracteres hexadecimales en el nombre del fichero
    hash_pattern = re.compile(r"^[0-9a-fA-F]+$")

    # Listado de hashes (sin la extensión .json)
    hashes = [
        p.stem
        for p in results_path.glob("*.json")
        if hash_pattern.match(p.stem)
    ]
    flat_records = []
    for h in hashes:
        with open(results_path / f"{h}.json", "r") as fres:
            res = json.load(fres)
        with open(metadata_path / f"{h}.json", "r") as fmeta:
            meta = json.load(fmeta)

        merged = {**meta, **res, "hash": h}
        flat = flatten_json(merged)
        flat_records.append(flat)
    combined_df = pd.DataFrame(flat_records)

    # 1) Build adjacency list: parent_hash → [child_hashes]
    tree = defaultdict(list)
    for _, row in combined_df.iterrows():
        h = row["hash"]
        prev = row.get("previous_hash")
        if pd.notna(prev):
            tree[prev].append(h)

    # 2) Find root nodes: those with no valid previous_hash
    all_hashes = set(combined_df["hash"])
    roots = [
        h
        for _, row in combined_df.iterrows()
        for h in [row["hash"]]
        if pd.isna(row.get("previous_hash"))
        or row["previous_hash"] not in all_hashes
    ]

    # 3) DFS to collect every root→leaf path
    experiments = []

    def _collect_paths(node, path):
        children = tree.get(node, [])
        if not children:
            experiments.append(path)
        else:
            for child in children:
                _collect_paths(child, path + [child])

    for root in roots:
        _collect_paths(root, [root])

    blacklist = {
        # ) hash                             # -> acc,    complexity,   type, bits, n_levels
        "82bcde2ecc786b41e6d2b33425c5295a",  # -> 0.6825, 2.522522e+06, flex, 8, 2
        "2d1d7bb39ddb4bce584d824a41b8aaa2",  # -> 0.6820, 2.346810e+06, flex, 8, 2
        "deb62ff4fda51063fb9094744ffc0051",  # -> 0.6803, 2.562207e+06, flex, 8, 2
        "8920925440a17cd21177b04319f4dde7",  # -> 0.6762, 2.629205e+06, flex, 8, 2
        "9eefaec10856b78c5b1b04cbbb2138ce",  # -> 0.6834, 5.739940e+06, flex, 4, 10
        "c6d6425d11b008bea5ae624601cf69f9",  # -> 0.6869, 8.300487e+06, flex, 6, 25
        "c24d8367f7eea4fe40c6b11d37a2ef64",  # -> 0.6829, 7.521903e+06, flex, 6, 30
        "f1a01952db947ca2a5dcf6932bc4dd9d",  # -> 0.6870, 7.761222e+06, flex, 6, 20
        "f76e8a717ae8f27c07edabd45a697813",  # -> 0.6803, 2.657772e+06, flex, 6, 2
        "318464ddf0bb2feb2249b20e4b2b0c35",  # -> 0.6802, 2.654884e+06, flex, 6, 3
        "a14dc019e8c2e2b8e4929ad2edae4d80",  # -> 0.6739, 2.905927e+06, flex, 6, 3
        "fe88a2a970c6d4fe95e1c5efb867de67",  # -> 0.6719, 4.990952e+06, flex, 6, 6
        "2d48e19bced1b6e3b15eab632a40cf44",  # -> 0.6682  2.476079e+06, flex, 4,    2
        "4965cf50af85c54e65044dc0b5717bcf",  # -> 0.6647  4.679128e+06, flex, 8,    5
        "cf82eb531d5c718017874750453af75e",  # -> 0.6622  5.459853e+06, flex, 4,    10
        "df2c27c6c58a72ce19f2bec69ba6fa22",  # -> 0.6602  4.127570e+06, flex, 4,    4
        "081eea60c304ff553e0ec04fcbfe17f8",  # -> 0.6566  8.184740e+06, flex, 6,    25
        "f8138453a8c78c1598af911967b22ef7",  # -> 0.6399  4.328394e+06, flex, 6,    5
        "0942ca5ce86afde3e272b8d3bb1c79cf",  # -> 0.6365  5.416113e+06, flex, 6,    7
        "57a410c7713212ae7e7874174a00ec28",  # -> 0.6358  5.352943e+06, flex, 4,    8
        "1ef38adee955e86d29015639c772708c",  # -> 0.6351  8.514102e+06, flex, 6,    30
        "ca9710fb7d998d349dc0fabffe3e7487",  # -> 0.6336  4.092541e+06, flex, 4,    5
        "f361e46f3f1a62d15f6ffe924f341e4b",  # -> 0.6294  5.502136e+06, flex, 6,    7
        "dcdc5e38399aa6b1f4aec6dd0f4a0011",  # -> 0.6228  5.499287e+06, flex, 6,    7
        "62293d44559425797da8b3cf5c03949d",  # -> 0.6226  5.282950e+06, flex, 8,    7
        "d3e21e49f4a6737a99c64ae7bd962ed9",  # -> 0.6173  8.716123e+06, flex, 6,    30
        "4b847613419cf65b6933ac7df186b5e1",  # -> 0.6172  4.120780e+06, flex, 6,    4
        "40c151e376a2a746a618bdea64acb645",  # -> 0.6118  8.659160e+06, flex, 8,    25
        "a1d029ad31e8d0e78a2727cb817238cd",  # -> 0.6100  9.312237e+06, flex, 8,    30
        "257185fcc8615ad18afa388080e0b42a",  # -> 0.6096  9.424802e+06, flex, 8,    30
        "cc7ee437a88cd218050280f5e1687d7b",  # -> 0.6091  4.411632e+06, flex, 4,    5
        "1c2b3e12e0e008acc93ad000dc46860c",  # -> 0.6085  8.087628e+06, flex, 8,    20
        "ad7f438d121a818e11559c02cf40e875",  # -> 0.6078  8.815668e+06, flex, 8,    25
        "bf3715b86ac5cece775d945518828380",  # -> 0.6075  8.649432e+06, flex, 6,    30
        "c671e4816625f3fcef3ae19735e588cb",  # -> 0.6065  8.057870e+06, flex, 6,    20
        "f77e1ff54c1e8548b16d78166e583a20",  # -> 0.6061  4.340742e+06, flex, 6,    5
        "f88ec612d8ea1bdcc25d3dc257d9078b",  # -> 0.6045  6.116025e+06, flex, 4,    10
        "cacd708264fa9f8700b9f4fb32b87494",  # -> 0.6013  5.497829e+06, flex, 4,    8
        "740a1562cfbc1d391c4f3a1cdbddd879",  # -> 0.5990  9.368490e+06, flex, 8,    30
        "18c96f0fc4d4aad8cef826c6cc048846",  # -> 0.5981  2.913844e+06, flex, 8,    3
        "6767ffdfcd17bf848370ba70b057aad9",  # -> 0.5969  8.600187e+06, flex, 6,    30
        "d8e9e5b7f38c90b8e4aa509d9b5ced31",  # -> 0.5961  5.227277e+06, flex, 4,    6
        "419c90ed7c09a63c3fc90a54661617b3",  # -> 0.5936  3.709586e+06, flex, 4,    5
        "5a9b0e4a09037ec75430ad220ccdde6d",  # -> 0.5930  8.775996e+06, flex, 8,    25
        "5d070cbf206198f489173a5f1ec74f91",  # -> 0.5927  4.441964e+06, flex, 8,    5
        "d36592e5b6e1894fba94f2264136580b",  # -> 0.5906  4.023145e+06, flex, 6,    4
        "6d9130bcab5126da553138f680ad5343",  # -> 0.5901  5.501902e+06, flex, 6,    7
        "228c22a86a8e5db74626432a4fa7ebf0",  # -> 0.5899  8.682319e+06, flex, 8,    25
        "ce99dd6e073611df6d56d0b035f2c380",  # -> 0.5894  5.523196e+06, flex, 8,    7
        "e4f9b423a715a006966cafd108626582",  # -> 0.5894  8.710177e+06, flex, 8,    25
        "dd44e1d79606bbf821f776b550fd5708",  # -> 0.5850  8.223336e+06, flex, 6,    20
        "553230a79d1600e5e96e4e55bced5562",  # -> 0.5847  4.876093e+06, flex, 4,    7
        "5b6c1d077fff399487198d0934d2f801",  # -> 0.5833  6.564747e+06, flex, 8,    10
        "a79306498306fd871b72f91886050764",  # -> 0.5833  4.978046e+06, flex, 4,    7
        "116279bfe5331dce42f5214acff1b8b8",  # -> 0.5827  4.314205e+06, flex, 6,    5
        "2eb0a2ba00ac93326abf3351bd1d9883",  # -> 0.5826  5.430122e+06, flex, 8,    7
        "f48060242ed1e5845835ec38f86f90e4",  # -> 0.5826  7.373076e+06, flex, 8,    16
        "a2a0c3db77ffc9e87502fce7370eb6a0",  # -> 0.5823  8.731863e+06, flex, 6,    25
        "ddc474f0d54a8cc5731b000baf77756d",  # -> 0.5809  9.161021e+06, flex, 8,    30
        "4126ec5dbf996778dcb49af8c4563a0e",  # -> 0.5790  4.420320e+06, flex, 8,    5
        "1dc469302d509868b5807c2c582781da",  # -> 0.5787  6.152794e+06, flex, 6,    10
        "a914f36d1b76973a6f5213a6cdb90e76",  # -> 0.5776  2.440996e+06, flex, 8,    3
        "07c097bac4abda58859eb5055fbb3443",  # -> 0.5759  5.499702e+06, flex, 4,    8
        "6351c9acd3391e61432ecfa4060a092a",  # -> 0.5756  4.607602e+06, flex, 8,    5
        "0828459655a6999d7d9af066c8632f60",  # -> 0.5751  5.512189e+06, flex, 8,    7
        "cfa18e91c2f434e01e9fa4d928bab7e4",  # -> 0.5749  2.745346e+06, flex, 6,    3
        "741cecf1ee1bbce01c1a2d3f5521c491",  # -> 0.5741  9.617448e+06, flex, 8,    30
        "81038213fd8189ba707569fb2b4da11c",  # -> 0.5734  5.008818e+06, flex, 8,    6
        "e913f582950807f522d9c802b6ef3f3e",  # -> 0.5719  4.106683e+06, flex, 8,    4
        "a47756b976c51cc11b50e1d8238c5200",  # -> 0.5716  4.141123e+06, flex, 6,    5
        "ed3fbf80c1e7546321a2dbd6ada1354d",  # -> 0.5715  7.853356e+06, flex, 8,    16
        "49befbd5aea7ec47387bd96665c38982",  # -> 0.5702  8.188566e+06, flex, 8,    20
        "253e6448f7f43101cbf863974d331d0a",  # -> 0.5685  8.324831e+06, flex, 8,    20
        "324fe5d7f47770902d73752a8e01d62d",  # -> 0.5663  4.599661e+06, flex, 4,    5
        "387929c287656e1418ecaca896f443ad",  # -> 0.5658  5.366461e+06, flex, 6,    7
        "724f9a6abede93b72c8fc7df30b8d9db",  # -> 0.5644  4.705492e+06, flex, 6,    6
        "446a454fc94b2789cd943578ed0bc074",  # -> 0.5640  4.995121e+06, flex, 4,    5
        "a7f630a537c947ccaf46387051efeef7",  # -> 0.5603  8.207445e+06, flex, 6,    20
        "2f23ee58330ffbafc3acb0b87a0699a4",  # -> 0.5593  9.103208e+06, flex, 8,    25
        "dbdc28ebce1b0a85029fce6714e49c9f",  # -> 0.5587  4.413465e+06, flex, 8,    5
        "b5358693b4337fc31be52a52df800328",  # -> 0.5583  5.549624e+06, flex, 6,    8
        "3feade8d67883ae66e406c0f6ac4cd57",  # -> 0.5579  4.478236e+06, flex, 4,    5
        "08b23ddf3b656f041d36152a88fba87e",  # -> 0.5574  5.441997e+06, flex, 6,    7
        "1b8cc4ea0c23fb930d198c9bace1c01f",  # -> 0.5565  4.984122e+06, flex, 4,    5
        "6982f70c319ed9df4e958de3b91baf23",  # -> 0.5564  2.965374e+06, flex, 6,    3
        "04ab8c630f0ac8a0935ccaa7297a19dd",  # -> 0.5536  8.794149e+06, flex, 6,    30
        "8412d76b73a091f1f075e3768f8c4485",  # -> 0.5526  8.143381e+06, flex, 8,    20
        "806a48aa4aead784c9dc93e64c4d694a",  # -> 0.5508  9.106848e+06, flex, 8,    25
        "f56b97ee2b9eb3673dde75b73acefa24",  # -> 0.5496  8.110230e+06, flex, 6,    20
        "f1a3c121b4c6331eee397599a8d98475",  # -> 0.5491  8.585863e+06, flex, 6,    25
        "53317e1a72893a4e61c5f07e1611a087",  # -> 0.5479  4.988744e+06, flex, 6,    6
        "18b39edfb145e5db5ce268793593fa55",  # -> 0.5478  8.271977e+06, flex, 8,    20
        "5dc8eb9e93379b1108d847486faec2b7",  # -> 0.5478  8.807598e+06, flex, 8,    30
        "30fc1daf89f19e980b5e1c69ea3eeddb",  # -> 0.5475  8.494731e+06, flex, 8,    20
        "efa5082ce34a54c5450944408dda3341",  # -> 0.5469  4.049373e+06, flex, 6,    5
        "2a57ca0c402c493d39a4fda0f722ef5f",  # -> 0.5464  3.183237e+06, flex, 4,    3
        "d878cba4575aa5c22fcd40084d9a3384",  # -> 0.5424  8.451002e+06, flex, 6,    25
        "30e350f38eebb57f58f88b01eb427b36",  # -> 0.5404  8.581810e+06, flex, 6,    25
        "12b856434f33147f95343017850b88f6",  # -> 0.5400  9.347335e+06, flex, 8,    30
        "91a5c093e1213c00cbf92e423795c04c",  # -> 0.5369  4.752921e+06, flex, 8,    6
        "a0ebf35a5577ba037ad4436360a8d631",  # -> 0.5336  8.461936e+06, flex, 6,    25
        "ed4e4713c7bcde4911563de5e51a6b1c",  # -> 0.5317  5.609182e+06, flex, 6,    7
        "d64d58115a0c16a93b46a26541ea940c",  # -> 0.5237  4.856726e+06, flex, 6,    6
        "33cc5a71d1e46710e53d58ae9e3fa7ec",  # -> 0.5228  8.773018e+06, flex, 8,    25
        "4751c5518326351fc7d5a2d1e20ae00b",  # -> 0.5224  9.387617e+06, flex, 8,    30
        "335513c169a9f64243badde924a39a8b",  # -> 0.5224  2.997398e+06, flex, 4,    3
        "2d07af1c3560853dfc51e2e95e76b42f",  # -> 0.5171  5.592623e+06, flex, 8,    8
        "37311b19ce9060557ab3c6fe57d2cb43",  # -> 0.5166  2.736840e+06, flex, 6,    2
        "f13028523f1d5bd0c69d624f1ee6c85d",  # -> 0.5116  3.993213e+06, flex, 6,    4
        "da98bfc8ea3478bb291828a5764d160c",  # -> 0.5106  4.432343e+06, flex, 8,    5
        "473516dffe62ea86713392c3cbc2fab9",  # -> 0.5092  5.323287e+06, flex, 8,    7
        "d5b85c422d79220101ba5f432ae51875",  # -> 0.5078  5.704372e+06, flex, 6,    8
        "b24210b8f23f81d5041f2ec4bf47741c",  # -> 0.5071  5.473601e+06, flex, 8,    7
        "1e1cc4492f0d9c28ba25c858687b311d",  # -> 0.5062  8.548573e+06, flex, 6,    25
        "a18788d22bdc7da6982159ae318ea1f9",  # -> 0.5056  2.637835e+06, flex, 4,    2
        "540fb846638db408db0a586efceec840",  # -> 0.5044  2.465916e+06, flex, 4,    2
        "1b8efa9a01c3246c1e4642e8aa65da03",  # -> 0.5031  6.534875e+06, flex, 6,    10
        "52fe89bc1409acc79ad0172460f6c46a",  # -> 0.5011  8.336156e+06, flex, 6,    25
        "c27ebd89aa6f73f30849fa1c190e6458",  # -> 0.5011  4.334441e+06, flex, 6,    5
        "adc2f6d7005c50c6237dc93869905211",  # -> 0.4982  8.812724e+06, flex, 6,    25
        "8ff65102493822dc4fd81ab19d25e033",  # -> 0.4962  6.517767e+06, flex, 8,    10
        "a45928491146a61ad7392180d03abf55",  # -> 0.4955  8.809505e+06, flex, 6,    30
        "e54464fe08d22fd6bc218ad716299cd0",  # -> 0.4940  8.181487e+06, flex, 6,    20
        "3ceb795754e7335c2a1f7a7d44eadd35",  # -> 0.4912  6.295182e+06, flex, 8,    10
        "6c2880840091d1caac5cd166141f2327",  # -> 0.4875  5.364793e+06, flex, 6,    7
        "98022e1420cb937e351850b1f56e96e6",  # -> 0.4868  4.514491e+06, flex, 8,    5
        "2cce305ea6540c328c79447cb79a39bd",  # -> 0.4805  2.725834e+06, flex, 8,    2
        "73cbba672e49d36f96fe434c00f5f3ab",  # -> 0.4789  5.761759e+06, flex, 8,    8
        "c2fe04aec32d0290c6c2a3b0bc7f1bbb",  # -> 0.4750  5.853028e+06, flex, 6,    8
        "3a1d5257cd243e9a896cfa0858b60450",  # -> 0.4721  4.908726e+06, flex, 8,    6
        "c3e8bd05639786207a3e49041c5dd5a1",  # -> 0.4717  5.403705e+06, flex, 6,    7
        "042c9b6b898e732c2531f960df4cc6b7",  # -> 0.4584  8.971624e+06, flex, 8,    25
        "d40576261b8f2060ad2f904f8380e4ed",  # -> 0.4571  8.335715e+06, flex, 8,    20
        "a498850d6b8debfcf5ae51d7d92aa488",  # -> 0.4567  5.613264e+06, flex, 4,    7
        "fe356e97e433dfa1a4a6daa0cb533d34",  # -> 0.4556  9.482259e+06, flex, 8,    30
        "09c8dd4e7f9645630608d2b335f4fd42",  # -> 0.4534  4.556333e+06, flex, 8,    5
        "4faa5b579f938a5a9a9ef2c287f54434",  # -> 0.4522  4.559044e+06, flex, 8,    5
        "00ae517067cbcd92ebfc024d843e480b",  # -> 0.4504  7.803484e+06, flex, 8,    16
        "96214e322683ecb99c0033cac1a86ed2",  # -> 0.4491  5.825073e+06, flex, 8,    8
        "7169526b3557e7f4567445ea2e1947a3",  # -> 0.4452  6.415053e+06, flex, 8,    10
        "47c7e524532935287485da7a54b9bbd0",  # -> 0.4451  8.338298e+06, flex, 6,    20
        "19bf11089cd1b489ad86b75937df27ec",  # -> 0.4447  7.706494e+06, flex, 8,    16
        "1f9c17051e51383b24ecb81a4bef1329",  # -> 0.4441  9.278590e+06, flex, 6,    30
        "e8ff6a262d0e6730fd6f55b141d9cfe4",  # -> 0.4364  5.390991e+06, flex, 8,    7
        "b7a9ae280c031ad7088b16bf178d809c",  # -> 0.4351  4.098193e+06, flex, 8,    4
        "fe0fd30a17fd52dba450c853536f5b27",  # -> 0.4318  2.739286e+06, flex, 6,    2
        "130b771d9ee96b22a28f67bd3f6651fe",  # -> 0.4305  5.336351e+06, flex, 8,    7
        "f2e3e46d510051393098cf95cd716f88",  # -> 0.4247  5.679510e+06, flex, 4,    7
        "41f7646a8bb31efbb89764cdadca2eb2",  # -> 0.4237  2.265290e+06, flex, 6,    3
        "cafe07776a80e8c84d020f0de156a195",  # -> 0.4188  5.431191e+06, flex, 8,    7
        "6784e09e4909dff9b28d740415f1ec50",  # -> 0.4179  4.478756e+06, flex, 8,    5
        "169a426b053dc7b9ece944dd1d83b578",  # -> 0.4148  9.194707e+06, flex, 6,    30
        "0abf9d5e11b9f3898898bae71bcf6e57",  # -> 0.4094  7.744605e+06, flex, 6,    16
        "e2498db25591d5e52723470a0953a725",  # -> 0.3957  9.452671e+06, flex, 8,    30
        "d92c79c0a81190c5cc6cfc4791cf70b0",  # -> 0.3940  5.638742e+06, flex, 4,    7
        "815f8290d3818251e4bd5a5bcb8acbd6",  # -> 0.3918  2.825758e+06, flex, 8,    3
        "416b6139ca6b68d874851170c7e349ac",  # -> 0.3868  4.204707e+06, flex, 4,    4
        "c9dc0f5cbde20b03fefe4aec336b2aa2",  # -> 0.3783  8.269713e+06, flex, 8,    20
        "0401977b495423cf305d2aa624340d36",  # -> 0.3770  6.460710e+06, flex, 6,    10
        "5c2ee969c48668d31e31d11cc78a491e",  # -> 0.3744  2.824383e+06, flex, 8,    2
        "f98adfa7a6daf49060b19966c4600e71",  # -> 0.3645  4.571965e+06, flex, 6,     5
        "3d96cfd6bb9ddb14f72a763a86ab8ca8",  # -> 0.3631  8.342203e+06, flex, 6,    20
        "54a35622869c5c9990669c443c529344",  # -> 0.3543  4.503398e+06, flex, 6,    5
        "77f800cbafb0f25cda2a0355d419b6d0",  # -> 0.3541  4.583585e+06, flex, 6,    5
        "176307c4bddce3f26248e089354bba0d",  # -> 0.3512  5.600199e+06, flex, 4,    7
        "37575e50ef4c718b4c7baab7a41f4735",  # -> 0.3509  4.371062e+06, flex, 8,    5
        "7889fe17fa9e15fbe7c872dc30c80d81",  # -> 0.3495  8.984966e+06, flex, 8,    25
        "1c0972c2b3b622621ff7ae3b223ee164",  # -> 0.3464  8.355320e+06, flex, 8,    20
        "25cce8bbc45c7f187800a9ce1220ed3a",  # -> 0.3413  5.552800e+06, flex, 8,    7
        "6a45374e3aeb7df392fd36e22cee190a",  # -> 0.3377  2.857394e+06, flex, 8,    2
        "217c00c772cb2980409890dede2b9a41",  # -> 0.3345  5.447785e+06, flex, 4,    7
        "42d8a1cc1266731836e3b2742469d8d8",  # -> 0.3250  7.750992e+06, flex, 6,    16
        "c7eb6de3357c58485a4111f365becb22",  # -> 0.3134  6.044683e+06, flex, 4,    10
        "71aa4790a9cdc70c8d86c511fe730540",  # -> 0.3071  8.114960e+06, flex, 8,    20
        "3bccd919a7bd3faeecb8f6165f0cb554",  # -> 0.3049  2.827845e+06, flex, 6,    2
        "00ad8a02b539bbcb40b4f9556d385d21",  # -> 0.2639  9.342941e+06, flex, 6,    30
        "48f8c6c676af256d5ac71f5124956b4a",  # -> 0.2589  5.430239e+06, flex, 8,    7
        "1e0e97bf6f7283f8822111174bf44875",  # -> 0.2543  4.367211e+06, flex, 6,    5
        "13b4e28a3c2adb02ebd26c6553322b77",  # -> 0.2511  4.777519e+06, flex, 4,    5
        "1dfb5b20677ca85df1e6e0029ed5ead3",  # -> 0.2366  8.310871e+06, flex, 6,    20
        "665f50e233c5e1ebfb99483684ee38b7",  # -> 0.2264  5.563939e+06, flex, 6,    7
        "cb80d372e2df52f2541a75f3e3ab6fa3",  # -> 0.1000  4.206023e+06, flex, 4,    6
        "4a38650f02cc8de4cbd89d3e3e811ff1",  # -> 0.1000  4.080933e+06, flex, 4,    6
        "9948a9597f24813d95a0bc76c153d0f8",  # -> 0.1000  3.957814e+06, flex, 4,    4
        "03d1bf347fc86923a4cd10126996565c",  # -> 0.1000  3.934121e+06, flex, 4,    6
        "342438e8ee1a31ea6a21f1d03c93c0e7",  # -> 0.1000  5.201594e+06, flex, 4,    7
        "3e797beb9ac8298f7826f65453fa3a2f",  # -> 0.1000  2.953513e+06, flex, 4,    2
        "367fb3948c6f69886ec3028a8360dc65",  # -> 0.6659, 4.209346e+06, flex, 6,    4
        "2e2ccc15724987b696acfdc21bf47a51",  # -> 0.6503, 7.269416e+06, flex, 6,    16
        "bc15239e6d2c175a0334f61af9848739",  # -> 0.6494, 5.148289e+06, flex, 4,    6
        "716f7759cf91aa844d7000045b1eb600",  # -> 0.6476, 2.741280e+06, flex, 8,    3
        "3c5f7b70f075fe8462c635153dada47d",  # -> 0.6303, 5.007099e+06, flex, 6,    6
        "8f0f60a3e18b7a24642cd76ed327b9a1",  # -> 0.6295, 5.785277e+06, flex, 6,    8
        "531d8b20ad7953e37f974f7d5b45e01c",  # -> 0.6267, 5.382755e+06, flex, 4,    8
        "57d1dc4ee01cee3cf0385670e1741322",  # -> 0.6211, 2.991667e+06, flex, 4,    3
        "05f70440f96aaf900cc613fb6a38610c",  # -> 0.6092, 6.469393e+06, flex, 6,    10
        "9b86d4910bd95fb6d262c5e5983c14ad",  # -> 0.6061, 4.349474e+06, flex, 4,    4
        "17b5fc2623904c1c66093c08e1b79f27",  # -> 0.6027, 6.106181e+06, flex, 4,    10
        "e67bd735524454b8a261787dcf2a47eb",  # -> 0.6000, 3.895603e+06, flex, 6,    4
        "fbe458505010a51dfc97863cfa4725ca",  # -> 0.5971, 6.166594e+06, flex, 4,    10
        "56c8a5c06806146647506acf32634294",  # -> 0.5963, 6.381624e+06, flex, 6,    10
        "0e8ab6e96cf978db32c2d7b7699cfe7f",  # -> 0.5901, 2.823915e+06, flex, 8,    3
        "f0cd6ddce67c86d57bf5661a26b5bde2",  # -> 0.5823, 4.080290e+06, flex, 8,    4
        "9e9f38b272165d42a0528ebdf9027f1c",  # -> 0.5815, 3.174854e+06, flex, 4,    3
        "4cc609a2bb8c5efa6791c32ea349850a",  # -> 0.5778, 5.486881e+06, flex, 4,    8
        "33eeff2627c8868d784639522b1eb4f1",  # -> 0.5747, 2.619111e+06, flex, 8,    3
        "7aec249371301c815c3eead824c6f7b1",  # -> 0.5738, 7.557755e+06, flex, 6,    16
        "abc9e0d7b65f1070bea9dfafa919f90f",  # -> 0.5725, 5.817821e+06, flex, 6,    8
        "90a9560d3a79b5ae23440476b04d9561",  # -> 0.5704, 6.553902e+06, flex, 8,    10
        "83973c46451c905ae0352fea31cd0df8",  # -> 0.5676, 7.421917e+06, flex, 8,    16
        "6a405a2cf4eb6ece8241296f29b52aad",  # -> 0.5637, 2.873761e+06, flex, 6,    3
        "bcaf7d4fe4b1a83c3d16efe2f6e4d967",  # -> 0.5615, 5.767932e+06, flex, 8,    8
        "d86af65d32bbb56f2eca4d54a8bdaeeb",  # -> 0.5529, 7.608056e+06, flex, 8,    16
        "a782864f124987e758d7911279c2b7cb",  # -> 0.5522, 2.935374e+06, flex, 6,    3
        "57a7e8afe58c32f72af8ea86cd18ddbb",  # -> 0.5511, 4.148061e+06, flex, 6,    4
        "0dbadd053f8161566f8bfd04e29f167b",  # -> 0.5500, 3.121372e+06, flex, 4,    3
        "03b75598bd4a4799823c4ac80dad72d1",  # -> 0.5483, 7.589383e+06, flex, 6,    16
        "77093fa86e3ac8b23d13065d8645d019",  # -> 0.5458, 6.490005e+06, flex, 8,    10
        "71d583b85d081e1e93bd75b836c83de6",  # -> 0.5402, 2.819020e+06, flex, 8,    2
        "36d5275416ed73b04c6952c4276c083d",  # -> 0.5292, 4.092483e+06, flex, 8,    4
        "b5bceb9302a3df3a2771deb7d0edd4c8",  # -> 0.5222, 7.549662e+06, flex, 8,    16
        "36d35f39f535e4aaa9cb5da438c8d903",  # -> 0.5154, 2.768778e+06, flex, 6,    2
        "a519ef258d54600ea6920c4788a6280b",  # -> 0.5086, 2.802222e+06, flex, 8,    2
        "513a1c16282dec9f124b9257bbb28177",  # -> 0.5075, 5.524247e+06, flex, 4,    8
        "8894cc71801b352192f5759cf887d6e4",  # -> 0.5066, 2.754725e+06, flex, 6,    2
        "1940550b5b9a07ce68968093723ef697",  # -> 0.5053, 6.330552e+06, flex, 8,    10
        "85444b910411afd5676269013fce39e6",  # -> 0.4950, 2.536435e+06, flex, 4,    2
        "1a704da76414aa5c50b9316544f50364",  # -> 0.4802, 2.524790e+06, flex, 4,    2
        "4b746c04189fe0077f7938802ac24dd4",  # -> 0.4729, 3.983884e+06, flex, 8,    4
        "7567c93fa1089f5cedc348292372ceb1",  # -> 0.4702, 2.734160e+06, flex, 8,    2
        "644bad7c98a5b9cb6985cb97d011c8f3",  # -> 0.4542, 4.932631e+06, flex, 8,    6
        "d38eaafd8b59ce694bd77c959b2c1e68",  # -> 0.4036, 5.795405e+06, flex, 8,    8
        "f99a35d5bbf685c7089d7519291b5c0f",  # -> 0.3773, 2.633205e+06, flex, 4,    2
        "ff48b26827e51e5a9e98b5e73ced1c98",  # -> 0.3749, 4.967528e+06, flex, 8,    6
        "b9815d8e206f5e7e7651761c998283e2",  # -> 0.3673, 5.903986e+06, flex, 6,    8
        "2e2c5d82931665495de2f92f4af6a095",  # -> 0.3622, 2.794733e+06, flex, 6,    2
        "dbdb4f07f3f92b592762aec1a402b14f",  # -> 0.3552, 2.728159e+06, flex, 6,    3
        "f33fc387a8c608886559ac95b3e4f7ce",  # -> 0.3458, 4.753085e+06, flex, 6,    6
        "66f7bfa092d65404f9a704e02c6e6f77",  # -> 0.3375, 4.760287e+06, flex, 8,    6
        "1af4b0115d6a08698fbfb4114a770489",  # -> 0.3331, 6.535710e+06, flex, 6,    10
        "0f3f929d0d9b0328f5e93ce11ea73b76",  # -> 0.1000, 3.965023e+06, flex, 4,    4
        "049dca947754753b7be1882fa3e5f0df",  # -> 0.1000, 3.978452e+06, flex, 4,    6
        "a8facc8acb1e1e2af5009a2d107a1c74",  # -> 0.1000, 4.142553e+06, flex, 4,    6
        "14ec0f67754688eb30f47f432595d4fa",  # -> 0.1000, 4.130849e+06, flex, 4,    4
        "478535a7bbb120ca3eee26c64c1a6fd6",  # -> 0.6697, 2.715756e+06, flex, 6,    2
        "9d3cdf362377b77f2769cddfc2f06cf2",  # -> 0.6670, 2.802477e+06, flex, 8,    3
        "8f70e90c6769dd3c9f44402acfec3606",  # -> 0.1000, 3.948141e+06, flex, 4,    6
        "58756c3729dcedfeeab0fea628634e05",  # -> 0.1000, 4.042809e+06, flex, 4,    4
        "ec39ccf1998a61cf388b6c64cf53e534",  # -> 0.1000, 4.040190e+06, flex, 4,    6
        "7c93ffb859c056bb8d58a57ea731a5b6",  # -> 0.1000, 3.956951e+06, flex, 4,    6
        "5c58106ae695dc7842cee914198213ce",  # -> 0.1000, 3.999102e+06, flex, 4,    4
        "66eb61bd460185ebb3a0e7e0903998d8",  # -> 0.1000, 3.915104e+06, flex, 4,    4
        "c26c75797cc5698524c128b2c03d71ad",  # -> 0.6699, 4.936011e+06, flex, 6,    6
        "1b5f11c64a41bebe6ecf86c89aec78e7",  # -> 0.6521, 2.760852e+06, flex, 6,    2
        "e8e38b72f113ae04a4b26c4c262d7fd2",  # -> 0.6504, 4.038943e+06, flex, 6,    4
        "936c04b9a6f211742285c459cc366134",  # -> 0.6487, 2.625230e+06, flex, 8,    2
        "654b17e55ed66715c84161e8f7dbec28",  # -> 0.1000, 4.271655e+06, flex, 4,    4
        "e5509196455a82fed7f194fd53f0e4c3",  # -> 0.1000, 3.883347e+06, flex, 4,    6
        "739b8f45159088a1cbfff30f92e29a59",  # -> 0.5141, 2.406360e+06, flex, 6,    3
        "fe398b8e7dfdc1cb8c71d2183a84de07",  # -> 0.1000, 3.996849e+06, flex, 4,    4
        "c8a4f191f560ea10931775d0a8320c08",  # -> 0.1000, 3.959836e+06, flex, 4,    6
        "04ccc689dc66e77aa8ecd339281c5ff0",  # -> 0.1000, 4.092232e+06, flex, 4,    6
        "bab2a48cbef368310b076926701822a3",  # -> 0.1000, 4.103027e+06, flex, 4,    4
    }
    experiments = [
        exp for exp in experiments if not any(h in blacklist for h in exp)
    ]

    print("After blacklisting, keeping", len(experiments), "experiments:")
    for exp in experiments:
        print(" → ".join(exp))

    # 1) build a lookup: hash → flat dict
    rec_map = {rec["hash"]: rec for rec in flat_records}

    # 2) for each experiment, merge the stages side-by-side, prefixing with the stage name
    experiments_data = []
    for exp in experiments:
        combined = {}
        for h in exp:
            rec = rec_map[h]
            stage = rec["name"]
            # prefix every field (including hash, loss, accuracy, nested flattened keys…)
            for col, val in rec.items():
                combined[f"{stage}_{col}"] = val
        experiments_data.append(combined)

    # 3) make a DataFrame
    experiments_df = pd.DataFrame(experiments_data)
    print(experiments_df.columns)
    print(
        experiments_df[
            [
                "qat_accuracy",
                "qat_complexity",
                "qat_hash",
                "quantization_parameters_kernel_1_type",
                "quantization_parameters_kernel_1_bits",
                "quantization_parameters_kernel_1_n_levels",
            ]
        ].sort_values(by=["qat_accuracy"], ascending=False)
    )

    to_drop = [
        "model_creation*",
        "freeze*",
        "*function",
        "*activations_*",
        "*bias*",
        "*input_shape*",
        "*_0_*",
        "*_2_*",
        "*_3_*",
        "*_name",
        "*_hash",
        "model_creation_seed",
        "quantization_seed",
        "qat_seed",
        "*dataset",
        "*_categories",
        "*_epochs",
        "*_batch_size",
        "*_learning_rate",
        "*_validation_split",
        "quantization_complexity",
        "qat_parameters_early_stopping",
    ]
    # find all columns matching any pattern
    cols_to_drop = [
        col
        for col in experiments_df.columns
        if any(fnmatch.fnmatch(col, pat) for pat in to_drop)
    ]
    experiments_df = experiments_df.drop(columns=cols_to_drop)

    rename_map = {
        "name": "stage",
        "quantization_parameters_kernel_1_type": "type",
        "quantization_parameters_kernel_1_bits": "bits",
        "quantization_parameters_kernel_1_n_levels": "n_levels",
        "initial_training_seed": "seed",
    }
    experiments_df = experiments_df.rename(columns=rename_map)
    experiments_df = experiments_df.dropna()

    experiments_df["qat_complexity"] = (
        experiments_df["qat_complexity"] / 1024
    )  # Convert to Kbits
    # print(
    #     experiments_df.sort_values(by=["qat_accuracy_mean", "qat_complexity_mean"], ascending=False)
    # )
    # print(experiments_df.sort_values(by=["qat_complexity_mean"], ascending=False))

    original_accuracy_mean = experiments_df["initial_training_accuracy"].mean()
    original_complexity_mean = (
        experiments_df["initial_training_complexity"].mean() / 1024
    )  # in Kbits
    original_accuracy_var = experiments_df["initial_training_accuracy"].var()
    original_complexity_var = experiments_df[
        "initial_training_complexity"
    ].var()
    original_accuracy_sd = np.sqrt(original_accuracy_var)
    original_complexity_var_kbits = original_complexity_var / (1024**2)
    original_complexity_sd = np.sqrt(original_complexity_var_kbits)
    n = experiments_df["initial_training_accuracy"].count()
    original_accuracy_se = original_accuracy_sd / np.sqrt(n)
    original_complexity_se = original_complexity_sd / np.sqrt(n)
    original_accuracy_min = experiments_df["initial_training_accuracy"].min()
    original_complexity_min = experiments_df[
        "initial_training_complexity"
    ].min()
    original_accuracy_max = experiments_df["initial_training_accuracy"].max()
    original_complexity_max = experiments_df[
        "initial_training_complexity"
    ].max()

    # 1) Identify the metrics and the hyperparam columns to group by
    metrics = ["qat_loss", "qat_accuracy", "qat_complexity"]
    hyperparam_cols = [
        col
        for col in experiments_df.columns
        if col in ("type", "bits", "n_levels")
    ]

    # 2) Group by those hyperparams, compute mean & var of the metrics
    stats = (
        experiments_df.groupby(hyperparam_cols)[metrics]
        .agg(["mean", "var", "max", "min"])
        .reset_index()
    )

    # 3) Flatten the resulting MultiIndex columns
    stats.columns = [
        f"{lvl0}_{lvl1}" if lvl1 else lvl0 for lvl0, lvl1 in stats.columns
    ]

    counts = (
        experiments_df.groupby(hyperparam_cols)[metrics].count().reset_index()
    )
    count_cols = [f"{m}_count" for m in metrics]
    counts.columns = hyperparam_cols + count_cols
    row_uniques = counts[count_cols].nunique(axis=1)
    assert (row_uniques == 1).all(), "Not all metric‐counts agree per group!"
    counts["count"] = counts[count_cols[0]]
    counts = counts[hyperparam_cols + ["count"]]
    stats = stats.merge(counts, on=hyperparam_cols, how="left")

    # 3) Compute standard deviations and errors
    stats["se_accuracy"] = np.sqrt(stats["qat_accuracy_var"]) / np.sqrt(
        stats["count"]
    )
    stats["se_complexity"] = np.sqrt(stats["qat_complexity_var"]) / np.sqrt(
        stats["count"]
    )

    print(
        stats.sort_values(
            by=["qat_accuracy_mean", "qat_complexity_mean"], ascending=False
        )
    )
    print(stats.sort_values(by=["qat_complexity_mean"], ascending=False))

    cmap = mpl.colormaps["tab10"]
    xmin = stats["qat_complexity_mean"].min() * 0.9
    xmax = stats["qat_complexity_mean"].max() * 1.1
    xmax = original_complexity_mean * 1.1

    plt.figure(figsize=(6, 4))
    plt.scatter(
        original_complexity_mean,
        original_accuracy_mean,
        color="red",
        label="Original Model",
        zorder=3,
    )
    bits_values = experiments_df["bits"].unique()
    for i, bits in enumerate(sorted(bits_values)):
        subset = experiments_df[experiments_df["bits"] == bits].sort_values(
            "qat_complexity"
        )
        # 1) dashed line only, semi-transparent
        plt.semilogx(
            subset["qat_complexity"],
            subset["qat_accuracy"],
            linestyle="--",
            color=cmap(i % 10),
            alpha=0.3,
            zorder=2,
            label=None,  # we’ll label in the marker call
        )
        # 2) opaque markers on top, with label
        plt.scatter(
            subset["qat_complexity"],
            subset["qat_accuracy"],
            marker=".",
            color=cmap(i % 10),
            zorder=3,
            label=f"{bits} bits",
        )
    add_horizontal_se_band(
        plt.gca(),
        original_accuracy_mean,
        original_accuracy_se,
        color="red",
    )
    plt.xlabel("Quantized Complexity (Kbits)")
    plt.ylabel("Quantized Accuracy")
    plt.ylim([0.6, 0.75])
    plt.xlim([xmin, xmax])
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_vs_quantized_flex.png", dpi=150)

    # 2) sort bits for consistent coloring
    plt.figure(figsize=(6, 4))
    plt.scatter(
        original_complexity_mean,
        original_accuracy_mean,
        color="red",
        label="Original Model",
        zorder=3,
    )
    bits_list = sorted(stats["bits"].unique())
    for i, bits in enumerate(bits_list):
        grp = stats[stats["bits"] == bits].sort_values("qat_complexity_mean")
        # 1) draw only the dashed line + errorbars (no markers)
        plt.errorbar(
            grp["qat_complexity_mean"],
            grp["qat_accuracy_mean"],
            xerr=grp["se_complexity"],
            yerr=grp["se_accuracy"],
            fmt="--",  # just the line
            color=cmap(i % 10),
            ecolor=cmap(i % 10),
            alpha=0.3,
            capsize=3,
            zorder=2,
        )
        # 2) draw the opaque markers on top
        plt.scatter(
            grp["qat_complexity_mean"],
            grp["qat_accuracy_mean"],
            marker=".",
            s=30,
            color=cmap(i % 10),
            label=f"{bits} bits",
            zorder=3,
        )
    add_horizontal_se_band(
        plt.gca(),
        original_accuracy_mean,
        original_accuracy_se,
        color="red",
    )
    plt.xscale("log")
    plt.xlabel("Quantized Complexity (Kbits)")
    plt.ylabel("Quantized Accuracy")
    plt.ylim([0.6, 0.75])
    plt.xlim([xmin, xmax])
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_vs_quantized_flex_stats.png", dpi=150)

    plt.figure(figsize=(6, 4))
    plt.scatter(
        original_complexity_mean,
        original_accuracy_mean,
        color="red",
        label="Original Model",
        zorder=3,
    )
    bits_list = sorted(stats["bits"].unique())
    for i, bits in enumerate(bits_list):
        grp = stats[stats["bits"] == bits].sort_values("qat_complexity_mean")
        plt.plot(
            grp["qat_complexity_mean"],
            grp["qat_accuracy_mean"],
            marker=".",
            linestyle="--",
            color=cmap(i % 10),
            label=f"{bits} bits",
            zorder=3,
        )
        plt.plot(
            grp["qat_complexity_mean"],
            grp["qat_accuracy_max"],
            linestyle="-",
            color=cmap(i % 10),
            alpha=0.5,
            zorder=2,
        )
        plt.plot(
            grp["qat_complexity_mean"],
            grp["qat_accuracy_min"],
            linestyle=":",
            color=cmap(i % 10),
            alpha=0.5,
            zorder=2,
        )
    add_horizontal_se_band(
        plt.gca(),
        original_accuracy_mean,
        original_accuracy_se,
        color="red",
    )
    plt.xscale("log")
    plt.xlabel("Quantized Complexity (Kbits)")
    plt.ylabel("Quantized Accuracy")
    plt.ylim([0.6, 0.75])
    plt.xlim([xmin, xmax])
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_vs_quantized_flex_stats2.png", dpi=150)
