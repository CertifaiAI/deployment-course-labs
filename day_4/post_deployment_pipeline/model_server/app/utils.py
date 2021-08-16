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

import logging
import sys
import pymongo

async def mongo_insert_doc(db, doc):
    doc['image_index'] = int(db.image_features_sequence.find_and_modify(
        query={ 'collection' : 'image_features' },
        update={'$inc': {'id': 1}},
        fields={'image_index': 0},
        new=True
    ).get('id'))

    try:
        db.image_features.insert(doc)

    except pymongo.errors.DuplicateKeyError as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise Exception(str(e))