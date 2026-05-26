Person of Interest(POI) Re-identification

Overview:
Real-time alert when an enrolled Person of Interest (POI) appears on any camera, and offline historical investigation of where a queried person appeared across all cameras and time ranges.
Functional Requirements
POI Creation
User should be able to upload one or more image by calling create poi api store the metadata details in redis with timestamp
Embeddings Generation
One user upload an image of poi post that we should create embeddings using face-detection-retail-0004 model and store the embeddings in the FAISS with a index poi all embiddings should use one poi index. 
Vector Storage
We will be using FAISS for embeddings storage because it’s fast and reliable.
Metadata storage
We will be using redis for metadata storage.
Object Detection & Tracking
When we receive new 
Matching Logic
The poi application must subscribe to one of the topiC of scenescape mqtt which will give message like below in continue streaming manner.
{
  "id": "bfb9f86b-b152-4e7f-8099-7c251ed84630",
  "debug_mac": "d7:88:c3:d6:4a:18",
  "timestamp": "2026-04-14T13:02:43.571Z",
  "debug_timestamp_end": "2026-04-14T13:02:43.857Z",
  "debug_processing_time": 0.28556060791015625,
  "rate": 10.091293507772983,
  "objects": [
    {
      "category": "person",
      "confidence": 0.9985646605491638,
      "center_of_mass": {
        "x": 846,
        "y": 232,
        "width": 52.666666666666664,
        "height": 89.75
      },
      "id": "50c4ec52-73b2-4686-b3ce-7b096840c198",
      "type": "person",
      "translation": [
        8.035146469023664,
        19.570617274265718,
        -8.728209237801713e-17
      ],
      "size": [
        1.0940080312538951,
        1.0940080312538951,
        1.731223725839966
      ],
      "velocity": [
        -0.13009222635324053,
        0.1424239487870952,
        0.0
      ],
      "rotation": [
        0,
        0,
        0,
        1
      ],
      "metadata": {
        "reid": {
          "embedding_vector": [
            [
              -0.31989994645118713,
              -0.2339610457420349,
              -0.08676224201917648,              
              0.2713760435581207
            ]
          ],
          "model_name": "torch-jit-export"
        }
      },
      "visibility": [
        "lp-camera1"
      ],
      "similarity": 44.57428741455078,
      "first_seen": "2026-04-14T13:02:36.176Z"
    },
    {
      "category": "person",
      "confidence": 0.6207658052444458,
      "center_of_mass": {
        "x": 1847,
        "y": 286,
        "width": 13.666666666666666,
        "height": 29.75
      },
      "id": "30088255-9fe7-4e45-8d13-2b34ae615284",
      "type": "person",
      "translation": [
        15.008401503938657,
        21.076045444246425,
        3.409249827069672e-16
      ],
      "size": [
        0.2961132845330692,
        0.2961132845330692,
        0.5680035291560153
      ],
      "velocity": [
        0.0005845705519019145,
        -0.0019982776603835258,
        0.0
      ],
      "rotation": [
        0,
        0,
        0,
        1
      ],
      "metadata": {
        "reid": {
          "embedding_vector": [
            [
              0.8027263879776001,
              -0.7955455780029297,
              -0.13804525136947632,
              0.6325618028640747,
              0.15575982630252838,
              0.40862149000167847
            ]
          ],
          "model_name": "torch-jit-export"
        }
      },  ],
  "debug_hmo_start_time": 1776171763.8600938,
  "name": "storewide loss prevention",
  "unique_detection_count": 36,
  "debug_hmo_processing_time": 0.003531932830810547
}

The embeddings should be extracted from above message then this should be search in the Faiss which we have stored in the poi index 

Store 
Alert Generation
Object Id level cashing
The system shall store all person movement events:
•	object_id 
•	timestamp 
•	region 
•	camera_id 
•	embedding reference (optional)
•	The system shall cache: 
object_id → poi_id
•	Subsequent frames for same object shall NOT trigger repeated FAISS searches

Session Tracking
API
Historical Search API
Create POI
http method: Post
payload : {images: [image1, image2], severity:1, description: string}
Response: 200
List POI
Send list of all the created poi with poi id in descending order by date
Delete POI
Delete poi by poi id
List Camera
This should return a list of cameras internally it will call scenescape camera api 
Here is the scenescape api which fetches camera list 
postman request 'https://<SCENESCAPE_API_URL>/api/v1/cameras' \
  --header 'Authorization: Token <SCENESCAPE_API_TOKEN>'
Historical Search api

This api request will have one image and start time and end time 
payload : {image:image, start_time:starttime, end_time: endtime}
Response:
person present in all the regions time and duration for all the regions should include region name also one thumbnail of user if present 

Configurations
There should be env and env.example file which should have below env variables
TAG = release tag

Scenescape configuration:
MQTT_HOST
MQTT_PORT
MQTT_TOPIC_EVENT
HTTP_PROXY = http proxy
HTTPS_PROXY = 
NO_PROXY
BENCHMARK_LATENCY
LOG_LEVEL
# Alert Service Environment Configuration

# ── MQTT Mode ────────────────────────────────────────────
MQTT_MODE=embedded

# ── MQTT Configuration ───────────────────────────────────
MQTT_HOST=
MQTT_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=

# Service Configuration
LOG_LEVEL=INFO
CONFIG_PATH=config/config.yaml

# Delivery Handlers Override
DELIVERY_HANDLERS=log,mqtt,websocket  

Docker
We need to build different image for ui which is in ui folder and has react based ui component
We need to build different image for poi backend which exist in backend folder 
Docker Compose 
Alert service: image -> intel/alert-service:0.0.1
Reids: image- > redis:8.6.2 or latest image as per application reuqireement
UI : poi application ui image
poi: poi application backend image

MAKE FILE
Make file should have below modes
make build REGISTRY=false : should build poi application locally 
make build : should build the application with images 
make up : should be calling docker compose up command and the application should be up
make down: application should be down
make logs : should show the logs across all services in application
make init-env : should initialize the env file with env example file with default values exist in example it if file exist it should ask it to override
make test : should run the pytest present in application
make coverage: should the pytest with coverage
make coverage-html: should run coverage also show the html file
make update-submodules: should update the githubs submodules from main branch example below
update-submodules:
    @echo "$(BLUE)Cloning/updating performance tool repositories...$(NC)"
    cd .. && git submodule deinit -f .
    cd .. && git submodule update --init --recursive --remote
    @echo "$(GREEN)Submodules updated successfully.$(NC)"


make help: should show all options available for make command with explanation
make restart: should restart all the services as well as application
make log-alert: should show the logs from alert service
make benchmark: 
make benchmark-stream-density
 
Benchmarking
Add performance tools as a gighub submodule in poi repository 
github link: https://github.com/intel-retail/performance-tools.git
We need to write two python script in the performance tools/benchmark-scripts, One script for single benchmark and another for benchmark stream density

Single Benchmark :
This can be run by calling make benchmark command 
Latency Calculation End to end: Time when person entered into the zone to the when we send an alert 
ReID_Inference_Latency_ms	Time taken by the ReID model to extract a 256-dim embedding vector from a single cropped person image
POI_Match_Latency_ms	Time from embedding extraction to FAISS gallery search result — how fast the system decides "is this a POI?
POI_Alert_Delivery_ms	Time between a confirmed POI match and the alert arriving at the security endpoint (REST/MQTT) — measures notification pipeline overhead

Benchmark Stream density



