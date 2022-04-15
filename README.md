# 설치
가급적 colab 환경(런타임 GPU 가속)으로 실행해주세요.

첫번째 cell은 사용할 라이브러리를 가져옵니다.

두번째 cell은 현재 github repository에 있는 파일을 가져옵니다.

모든 파일을 성공적으로 불러오지 못할 경우, 아래의 '알려진 문제'를 확인해 주세요
```python
!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
!sudo apt-get install git-lfs
!git lfs install
!git clone https://github.com/Jaeyoon-Park/project01
%cd project01
```
# 사용법
## OpenCV ver.
### Data Preprocess
헤드 어택 이미지 Mask를 만들기 위해 HSV 영역의 Histogram을 확인합니다.
```python
img = cv2.imread("head attack sample.jpg")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hist_h = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
hist_s = cv2.calcHist([img_hsv], [1], None, [255], [0, 255])
hist_v = cv2.calcHist([img_hsv], [2], None, [255], [0, 255])

plt.plot(hist_h) # blue line 
plt.plot(hist_s) # orange line
plt.plot(hist_v) # green line
plt.show()
```
HSV 범위를 설정합니다. 기본 설정으로 진행할 경우, 아래의 cell로 이동하여 실행해주세요.
```python
# 아래의 범위에서 "헤드어택" 문구가 제일 잘 보임
mask_hsv = cv2.inRange(img_hsv, (0, 15, 220), (40, 155, 255))
cv2_imshow(mask_hsv)
```
테스트하고자 하는 영상을 불러옵니다.
문제없이 불러왔을 경우, 영상의 크기와 프레임수, fps 정보를 확인할 수 있습니다.
```python
capture = cv2.VideoCapture("lostark sample.mp4")

capture_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
capture_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
capture_total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

fps = capture.get(cv2.CAP_PROP_FPS)

print("{0} x {1}".format(capture_w, capture_h))
print(capture_total_frame)
print(fps)
```
스킬을 사용했는지 확인하려면 기준 이미지가 필요합니다.

슬라이더를 사용하여 영상의 프레임을 조정하고,

cv2_imshow를 통해 스킬이 모두 기본 상태(사용 전 상태)인지 확인합니다.
```python
# 슬라이더로 프레임 설정
import ipywidgets as widgets
slider = widgets.IntSlider(value=5, max=capture_total_frame)
display(slider)
# 슬라이더 값으로 비디오 프레임 조정 및 화면 확인
capture.set(cv2.CAP_PROP_POS_FRAMES, slider.value)
_, bg = capture.read()
cv2_imshow(bg)
```
'헤드어택 성공 비율을 확인하려는 스킬 선택' 제목의 cell을 실행하여

헤드 어택 성공율을 확인하고자 하는 스킬을 입력합니다.
### Processing
'해당 스킬 사용 횟수 및 프레임 확인' 제목의 cell을 실행하면

입력한 스킬을 사용한 프레임이 결과창에 나타납니다.

앞서 설정한 기준 이미지 대비 스킬 이미지의 차이만을 기준으로 삼으면

스킬의 쿨타임인 순간에도 스킬을 사용했다고 중복으로 횟수를 체크합니다.

이를 방지하기 위해 각 프레임마다 차이를 계산하여

차이의 변화량이 커질 경우를 스킬 사용으로 체크합니다. (threshold 70)

```python
skill_use = cv2.absdiff(roi, roi_bg)
mean_value_skill_use = float(cv2.mean(skill_use)[0])
skill_use_frame_mean_list.append(mean_value_skill_use)
skill_used = abs(skill_use_frame_mean_list[i-1]-skill_use_frame_mean_list[i])
```
하지만 컷신이 발생할 경우에도 스킬 이미지 roi 영역의 변화가 있기 때문에 이를 방지하기 위해

컷신이 아니면 변하지 않는 영역을 설정하여, 변화 threshold인 30 이상일 경우를 제외했습니다.
```python
# 스킬창 roi 변화량과 상단 메뉴바 roi 변화량 기준으로 스킬 사용 여부 체크
if skill_used > 70 and cutscene < 30:
  skill_use_frame_idx_list.append(int(capture.get(cv2.CAP_PROP_POS_FRAMES)))
  cv2_imshow(frame)
```
'스킬 사용 당시 헤드어택 발생 여부 확인 (Template Matching)' 제목의 cell을 실행합니다.

스킬을 사용한 프레임 이후부터 하나씩 '헤드 어택'과 일치하는 모양이 있는지 검사합니다.

검출되면 '헤드 어택'으로 인식한 곳에 빨간 box가 생성된 이미지가 결과창에 출력됩니다.
### Result
상기 입력 스킬의 사용 횟수와 헤드어택 횟수를 출력합니다. 
## YOLOv5 ver.
'미리 custom data로 학습된 가중치 파일 best.pt 로드하기' 제목의 cell을 실행합니다.
```python
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
```
경로 내에 best.pt가 없거나, 새로운 가중치 파일을 사용하고 싶을 경우,

링크(https://github.com/ultralytics/yolov5) 를 이용해 custom training을 진행해주세요.

이후 순서대로 cell을 실행합니다.

자세한 내용은 OpenCV ver.의 테스트하고자 하는 영상을 불러오는 부분을 참고하세요.
### Processing
'해당 스킬 사용 횟수 및 프레임 확인' 제목의 cell을 실행하면

입력한 스킬을 사용한 프레임이 결과창에 나타납니다. OpenCV ver. 와 동일하므로

자세한 내용은 OpenCV ver.의 Processing 부분을 참고해주세요.

'스킬 사용 당시 헤드어택 발생 여부 확인 (custom YOLOv5 모델 적용)' 제목의 cell을 실행하면

스킬을 사용한 후, 헤드 어택 객체가 검출되었는지 확인합니다.

검출 결과는 ./runs/detect/exp 폴더에 저장됩니다.
### Result
상기 입력 스킬의 사용 횟수와 헤드어택 횟수를 출력합니다. 
# 알려진 문제
git clone 시, git-lfs 사용으로 인해 bandwidth 초과시 정상적으로 작동하지 않는 문제

(해당 이슈 발생시 파일의 2번째 cell을 아래와 같이 변경하고, git 전체 파일을 google 드라이브에 업로드해주세요.)
```python
from google.colab import drive
import os
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive')
```
