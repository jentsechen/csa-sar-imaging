import boto3
from botocore import UNSIGNED
from botocore.config import Config
import json
import folium

def mark_scene_center():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'umbra-open-data-catalog'
    base_prefix = 'sar-data/tasks/ship_detection_testdata/'
    response = s3.list_objects_v2(Bucket=bucket, Prefix=base_prefix, Delimiter='/')
    
    lla_uuid_list = []
    for uuid_prefix in response['CommonPrefixes']:
        uuid = uuid_prefix['Prefix'].replace(base_prefix, "").strip('/')
        folder_prefix = f"{base_prefix}{uuid}/"
            
        response = s3.list_objects_v2(Bucket=bucket, Prefix=folder_prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].lower().endswith('.json'):
                    content_object = s3.get_object(Bucket=bucket, Key=obj['Key'])
                    json_data = json.loads(content_object['Body'].read().decode('utf-8'))
                    
                    # --- MODIFIED LOGIC START ---
                    # Umbra metadata stores geometric info inside the 'collects' list
                    json_value = "Key Not Found"
                    
                    if 'collects' in json_data and len(json_data['collects']) > 0:
                        collect = json_data['collects'][0]
                        if 'sceneCenterPointLla' in collect:
                            json_value = collect['sceneCenterPointLla']
                    # --- MODIFIED LOGIC END ---

                    print(f"Processed {uuid}: {json_value}")
                    lla_uuid = json_value['coordinates']
                    lla_uuid.append(uuid)
                    lla_uuid_list.append(lla_uuid)
                    break
    map_center = [lla_uuid_list[0][1], lla_uuid_list[0][0]]
    m = folium.Map(location=map_center, zoom_start=12)
    for lla_uuid in lla_uuid_list:
        lon, lat, _, uuid = lla_uuid
        folium.Marker(
            location=[lat, lon],
            popup=f"{uuid}",
            # tooltip="SAR Scene Center"
        ).add_to(m)
    m.save("map.html")
    print("Map saved as map.html")

if __name__ == "__main__":
    mark_scene_center()