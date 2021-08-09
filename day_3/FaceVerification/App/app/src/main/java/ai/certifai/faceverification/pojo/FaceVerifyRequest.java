/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
 * This program is part of OSRFramework. You can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version. You should have received a copy of the
 * GNU Affero General Public License along with this program.  If not, see
 * https://www.gnu.org/licenses/agpl-3.0
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 */

package ai.certifai.faceverification.pojo;

import com.google.gson.annotations.SerializedName;

/**
 * Face verification api request body
 *
 * @author willardsm
 */
public class FaceVerifyRequest
{
    @SerializedName("image1_string")
    private String image1_string;

    @SerializedName("image2_string")
    private String image2_string;

    public FaceVerifyRequest(String image1_string, String image2_string) {
        this.image1_string = image1_string;
        this.image2_string = image2_string;
    }

    public String getImage1_string() {
        return image1_string;
    }

    public void setImage1_string(String image1_string) {
        this.image1_string = image1_string;
    }

    public String getImage2_string() {
        return image2_string;
    }

    public void setImage2_string(String image2_string) {
        this.image2_string = image2_string;
    }
}
