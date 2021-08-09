#  Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
#  This program is part of OSRFramework. You can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version. You should have received a copy of the
#  GNU Affero General Public License along with this program.  If not, see
#  https://www.gnu.org/licenses/agpl-3.0
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Affero General Public License for more details.

"""
python script to download YOLOv5 model
prepared by: YCCertifai
"""

import os
from pathlib import Path
import wget


def download(source, target, filename):
    if not os.path.exists(target):
        os.mkdir(target)
    targt_file = str(Path(target).joinpath(filename))
    if os.path.exists(targt_file):
        print('data already exists, skipping download')
        return
    print("Downloading from {} to {}".format(source, target))
    wget.download(source, targt_file)
    print("\nDone!")


if __name__ == "__main__":
    source = 'https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/model.pt'
    target = '../App/app/src/main/assets/'
    filename = 'model.pt'
    download(source, target, filename)
