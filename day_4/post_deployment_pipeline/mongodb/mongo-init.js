/**
 * 
 * Copyright (c) 2021 CertifAI Sdn. Bhd.
 *
 * This program is part of OSRFramework. You can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

var rootUser = 'root';
var rootPassword = 'admin';
var user = 'admin';
var passwd = 'admin';

var admin = db.getSiblingDB('drift_detection');

admin.auth(rootUser, rootPassword);

db.createUser({
  user: user,
  pwd: passwd,
  roles: [
    {
      role: "readWrite",
      db: "drift_detection"
    }
  ]
});

db.createCollection('image_features');

// this is for auto-increment in image_index column in image_features collection
db.image_features_sequence.insert({
  'collection': 'image_features',
  'id': 0
});;

// CREATE ANOTHER SET OF COLLECTIONS FOR ALTERNATIVE MODEL
db.createCollection('image_features_new');
db.image_features_sequence_new.insert({
  'collection': 'image_features_new',
  'id': 0
});;