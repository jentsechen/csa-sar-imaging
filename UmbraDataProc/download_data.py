import boto3
from botocore import UNSIGNED
from botocore.config import Config
import json
import folium

def list_data_size():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'umbra-open-data-catalog'
    prefix = 'sar-data/tasks/ship_detection_testdata/'
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')

    with open("data_size_list.txt", "w", encoding="utf-8") as file:
        for uuid_prefix in response['CommonPrefixes']:
            current_uuid = uuid_prefix['Prefix'].replace(prefix, "").strip('/')
            files_in_uuid = s3.list_objects_v2(Bucket=bucket, Prefix=uuid_prefix['Prefix'])
            
            if 'Contents' in files_in_uuid:
                for obj in files_in_uuid['Contents']:
                    if obj['Key'].lower().endswith('.cphd'):
                        file_name = obj['Key'].split('/')[-1]
                        size_gb = obj['Size'] / (1024**3)
                        file.write(f"{current_uuid:<40} | {file_name:<40} | {size_gb:>5.2f} GB\n")

def find_missing_uuids(uuid_list_file, data_size_file):
    # 1. Read all UUIDs from the master list
    with open(uuid_list_file, 'r', encoding='utf-8') as f:
        # Strip whitespace/newlines from each line
        master_uuids = set(line.strip() for line in f if line.strip())

    # 2. Read the UUIDs already processed in data_size.txt
    processed_uuids = set()
    with open(data_size_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                # Extract the first column (the UUID) before the first pipe
                uuid_part = line.split('|')[0].strip()
                processed_uuids.add(uuid_part)

    # 3. Find the difference (Items in master but NOT in processed)
    missing_uuids = master_uuids - processed_uuids

    # 4. Output the results
    print(f"Total UUIDs in master list: {len(master_uuids)}")
    print(f"Total UUIDs in data_size:  {len(processed_uuids)}")
    print(f"Missing UUIDs:             {len(missing_uuids)}")
    print("-" * 40)
    
    for uuid in sorted(missing_uuids):
        print(uuid)

def extract_metadata_from_list(input_file, target_json_key):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'umbra-open-data-catalog'
    base_prefix = 'sar-data/tasks/ship_detection_testdata/'
    
    lla_uuid_list = []
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"Total lines found in file: {len(lines)}")
        for line in lines:
            if "|" not in line: continue
            
            uuid = line.split('|')[0].strip()
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
                            if target_json_key in collect:
                                json_value = collect[target_json_key]
                        # --- MODIFIED LOGIC END ---

                        print(f"Processed {uuid}: {json_value}")
                        lla_uuid = json_value['coordinates']
                        lla_uuid.append(uuid)
                        lla_uuid_list.append(lla_uuid)
                        break
            # if len(lla_uuid_list)==10:
            #     break
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
    # list_data_size()
    # find_missing_uuids('uuid_list.txt', 'data_size.txt')
    # extract_metadata_from_list(input_file='data_size_list.txt', target_json_key='sceneCenterPointLla')
    mark_scene_center()