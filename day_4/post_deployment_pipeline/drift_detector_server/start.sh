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

# runs 2 commands simultaneously:

# first application
# Uncomment line with localhost when testing
uvicorn app.main:app --host 0.0.0.0 --port 8009 & 
# uvicorn app.main:app --host localhost --port 8009 & 
P1=$!

# second application
python scheduler/pull_mongo_data.py & 
P2=$!
wait $P1 $P2