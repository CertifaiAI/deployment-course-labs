#
#################################################################################
#
#  Copyright (c) 2021 CertifAI Sdn. Bhd.
#
#  This program is part of OSRFramework. You can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#################################################################################
#
from app.config import NUM_CLASSES, DRIFT_DETECTOR_PATH

import pickle

class DriftDetector:
    def __init__(self):
        self.drift_detector = pickle.load(open(DRIFT_DETECTOR_PATH,'rb'))

    def detect(self, image_features):
        # compute test score (loss)
        score = self.drift_detector(image_features)
        
        # compute p-value
        p_val = self.drift_detector.compute_p_value(image_features)

        return score.item(), p_val.item()
