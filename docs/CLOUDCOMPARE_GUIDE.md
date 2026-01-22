# CloudCompare 부재 좌표 추출 가이드

실제 비계 점군에서 각 부재의 Bounding Box 좌표를 추출하여 CSV 파일을 만드는 상세 가이드입니다.

---

## 준비물

- **CloudCompare** (무료): https://cloudcompare.org/release/
- **비계 점군 파일**: .ply, .las, .pcd, .xyz 등
- **메모장 또는 Excel**: CSV 작성용

---

## 최종 목표: 이런 CSV 파일 만들기

**권장 형식 (Center/Dimension) - CloudCompare 값 그대로 복사:**
```csv
component_id,type,cx,cy,cz,dx,dy,dz
c1,vertical,-5.937477,5.495575,116.242111,1.31834,0.200557,0.140015
c2,horizontal,0.5,3.2,112.5,10.5,0.15,0.08
c3,platform,0.0,2.5,115.0,2.0,1.5,0.05
```

CloudCompare에서 보이는 값:
- `cx, cy, cz` ← Global Box Center의 X, Y, Z
- `dx, dy, dz` ← Box Dimensions의 X, Y, Z

**대안 형식 (Min/Max) - 직접 계산한 경우:**
```csv
component_id,type,x_min,x_max,y_min,y_max,z_min,z_max
c1,vertical,-6.60,-5.28,5.40,5.60,116.17,116.31
```

---

## Step 1: CloudCompare 실행 및 점군 열기

### 1.1 CloudCompare 실행
- Windows: 시작 메뉴 → CloudCompare
- Mac: Applications → CloudCompare

### 1.2 점군 파일 열기
```
메뉴: File → Open
또는: Ctrl+O (Windows) / Cmd+O (Mac)
또는: 파일을 CloudCompare 창에 드래그 앤 드롭
```

### 1.3 파일 선택
- 비계 점군 파일 선택 (.ply, .las, .pcd 등)
- "Open" 클릭

### 1.4 Import 옵션 (팝업이 뜨면)
- 대부분 기본값 유지
- "Apply" 또는 "OK" 클릭

### 1.5 확인
- 왼쪽 **DB Tree** 패널에 점군 이름이 나타남
- 3D 뷰에 점군이 표시됨

---

## Step 2: 화면 조작 익히기

### 마우스 조작
| 동작 | 기능 |
|------|------|
| 왼쪽 드래그 | 회전 |
| 오른쪽 드래그 | 이동 (팬) |
| 스크롤 휠 | 확대/축소 |
| 휠 클릭 드래그 | 이동 (팬) |

### 유용한 단축키
| 단축키 | 기능 |
|--------|------|
| `1` | 정면 뷰 (Front) |
| `3` | 오른쪽 뷰 (Right) |
| `7` | 위에서 뷰 (Top) |
| `5` | 직교/원근 전환 |
| `Z` | 전체 화면에 맞추기 |

---

## Step 3: CSV 파일 준비

### 3.1 Excel 또는 메모장 열기

### 3.2 첫 번째 줄 (헤더) 작성
```
component_id,type,x_min,x_max,y_min,y_max,z_min,z_max
```

### 3.3 파일 저장 준비
- Excel: 나중에 "CSV (쉼표로 분리)" 형식으로 저장
- 메모장: .csv 확장자로 저장

---

## Step 4: 부재 선택 (Segment Tool) - 핵심!

### 4.1 Segment Tool 활성화

**방법 A: 툴바 사용**
```
툴바에서 가위(✂️) 아이콘 클릭
- 위치: 상단 툴바 중앙 부근
- 아이콘 모양: 가위 또는 다각형 모양
```

**방법 B: 메뉴 사용**
```
메뉴: Edit → Segment
```

**방법 C: 단축키**
```
단축키: 없음 (툴바나 메뉴 사용)
```

### 4.2 Segment Tool 활성화 확인
- 화면 상단에 Segment 툴바가 나타남:
  ```
  [Pause] [Segment In] [Segment Out] [✓ Confirm] [✗ Cancel] [Undo] [Clear]
  ```

---

## Step 5: 부재 영역 선택하기

### 5.1 부재가 잘 보이도록 화면 조정
```
1. 마우스로 회전하여 선택할 부재가 잘 보이게 함
2. 스크롤로 확대
3. 다른 부재와 겹치지 않는 각도 찾기
```

### 5.2 다각형으로 부재 영역 그리기
```
1. 부재 주변에서 왼쪽 클릭 → 첫 번째 점
2. 부재를 따라 이동하며 왼쪽 클릭 → 점 추가
3. 부재를 완전히 감싸도록 여러 점 클릭 (4-8개 정도)
4. 오른쪽 클릭 → 다각형 완성 (닫힘)
```

**주의사항:**
- 부재를 완전히 감싸야 함
- 다른 부재가 포함되지 않도록 주의
- 약간 여유있게 선택해도 괜찮음

### 5.3 선택 영역 확정
```
"Segment In" 버튼 클릭
- 다각형 안쪽 점들만 남음
- 나머지는 숨겨짐
```

### 5.4 선택 확정
```
체크마크(✓) "Confirm segmentation" 클릭
```

### 5.5 결과 확인
- DB Tree에 새로운 점군이 생성됨:
  - `[원본 이름].remaining` - 선택되지 않은 부분
  - `[원본 이름].part` - 선택된 부분 (이게 부재!)

---

## Step 6: Bounding Box 좌표 확인 ⭐ (가장 중요!)

### 6.1 선택된 부재 클릭
```
DB Tree에서 "[원본].part" 클릭하여 선택
```

### 6.2 Properties 패널에서 값 확인

Properties 패널 (왼쪽 하단)에서 다음 값들을 찾습니다:

```
Box dimensions:        ← dx, dy, dz 값
  X: 1.31834
  Y: 0.200557
  Z: 0.140015

Global box center:     ← cx, cy, cz 값
  X: -5.937477
  Y: 5.495575
  Z: 116.242111
```

### 6.3 CSV에 바로 기록 (권장 방식)

위 값을 그대로 복사해서 CSV에 기록:

```csv
component_id,type,cx,cy,cz,dx,dy,dz
c1,vertical,-5.937477,5.495575,116.242111,1.31834,0.200557,0.140015
```

**각 값의 의미:**
| CloudCompare 항목 | CSV 컬럼 | 설명 |
|------------------|----------|------|
| Global box center X | cx | 중심 X 좌표 |
| Global box center Y | cy | 중심 Y 좌표 |
| Global box center Z | cz | 중심 Z 좌표 |
| Box dimensions X | dx | X 방향 크기 (폭) |
| Box dimensions Y | dy | Y 방향 크기 (깊이) |
| Box dimensions Z | dz | Z 방향 크기 (높이) |

### 6.4 (대안) Min/Max 형식으로 기록

직접 계산하고 싶다면:
```
x_min = cx - dx/2 = -5.937477 - 1.31834/2 = -6.596647
x_max = cx + dx/2 = -5.937477 + 1.31834/2 = -5.278307
... (y, z도 동일)
```

```csv
component_id,type,x_min,x_max,y_min,y_max,z_min,z_max
c1,vertical,-6.596647,-5.278307,5.395297,5.595854,116.172104,116.312119
```

---

## Step 7: CSV에 기록

### 7.1 부재 정보 기록

Excel 또는 메모장에 한 줄 추가 (Center/Dimension 형식):
```
c1,vertical,-5.937477,5.495575,116.242111,1.31834,0.200557,0.140015
```

순서: `component_id, type, cx, cy, cz, dx, dy, dz`

### 7.2 부재 유형 판단

**Vertical Post (수직 기둥)**
```
특징:
- Z 범위가 큼 (높이가 높음)
- X, Y 범위가 작음 (단면이 작음)

예: z_max - z_min = 24.567 (높이 약 24m)
    x_max - x_min = 0.1 (폭 약 10cm)
```

**Horizontal Beam (수평 빔)**
```
특징:
- X 또는 Y 방향으로 길쭉함
- Z 범위가 작음 (두께만)

예: x_max - x_min = 10.0 (길이 10m)
    z_max - z_min = 0.05 (두께 5cm)
```

**Platform (발판)**
```
특징:
- X, Y 모두 넓음 (평평한 판)
- Z 범위가 작음 (두께만)

예: x_max - x_min = 2.0
    y_max - y_min = 0.8
    z_max - z_min = 0.04 (두께 4cm)
```

---

## Step 8: 원본 점군 복구 및 다음 부재 선택

### 8.1 현재 선택 취소/삭제
```
방법 A: Segment 된 부분 삭제
1. DB Tree에서 ".part" 와 ".remaining" 둘 다 선택 (Ctrl+클릭)
2. Delete 키 또는 우클릭 → Delete

방법 B: Undo
1. Ctrl+Z 여러 번
```

### 8.2 원본 다시 표시
```
1. DB Tree에서 원본 점군 찾기
2. 체크박스가 해제되어 있으면 체크하여 표시
```

### 8.3 다음 부재 선택
```
Step 4-7 반복
```

---

## Step 9: 모든 부재 완료 후 CSV 저장

### 9.1 최종 CSV 내용 예시 (Center/Dimension 형식)
```csv
component_id,type,cx,cy,cz,dx,dy,dz
c1,vertical,-5.937477,5.495575,116.242111,1.31834,0.200557,0.140015
c2,vertical,4.590,5.577,116.242,1.312,0.201,0.140
c3,vertical,-5.702,8.555,116.242,1.312,0.201,0.140
c4,horizontal,0.500,3.200,112.500,10.500,0.150,0.080
c5,horizontal,0.500,3.200,118.500,10.500,0.150,0.080
c6,platform,0.000,2.500,115.000,2.000,1.500,0.050
c7,platform,0.000,2.500,121.000,2.000,1.500,0.050
```

### 9.2 파일 저장
```
Excel: 파일 → 다른 이름으로 저장 → CSV (쉼표로 분리)(*.csv)
메모장: 파일 → 저장 → 파일명.csv
```

---

## 빠른 작업을 위한 팁

### 팁 1: 뷰 저장
```
자주 쓰는 시점을 저장:
Display → Save viewport as object
```

### 팁 2: 점 크기 조절
```
점이 너무 작거나 크면:
1. DB Tree에서 점군 선택
2. Properties → Point size 조절
```

### 팁 3: 색상 변경
```
부재 구분을 위해:
1. 점군 선택
2. Edit → Colors → Set unique color
```

### 팁 4: 부재별로 저장
```
나중에 확인용으로 각 부재를 별도 파일로 저장:
1. 부재 점군 선택
2. File → Save
3. c1_vertical.ply 등으로 저장
```

---

## 문제 해결

### 문제: Segment 다각형이 안 그려짐
```
해결:
1. DB Tree에서 점군이 선택되어 있는지 확인
2. 점군 옆의 체크박스가 켜져 있는지 확인
3. Segment Tool을 다시 활성화
```

### 문제: Bounding Box가 안 보임
```
해결:
1. Properties 패널이 보이는지 확인
   - 안 보이면: Display → Properties 또는 Ctrl+P
2. Properties 패널을 스크롤하여 Bounding Box 섹션 찾기
3. 또는: Edit → Compute → Bounding Box 실행
```

### 문제: 좌표값이 이상하게 큼 (예: 12345678.xxx)
```
원인: 점군이 실제 좌표계 (GPS 등) 사용
해결: 그대로 사용해도 됨 (코드가 자동 정규화)
```

### 문제: 부재 선택 시 다른 부재도 포함됨
```
해결:
1. 화면을 회전하여 부재가 분리되어 보이는 각도 찾기
2. 더 정밀하게 다각형 그리기
3. 약간 포함되어도 큰 문제 없음 (bbox 범위만 약간 넓어짐)
```

---

## 체크리스트

작업 전:
- [ ] CloudCompare 설치됨
- [ ] 점군 파일 준비됨
- [ ] CSV 파일 헤더 작성됨: `component_id,type,cx,cy,cz,dx,dy,dz`

각 부재마다:
- [ ] Segment Tool로 부재 선택
- [ ] Segment In → Confirm
- [ ] Properties에서 확인:
  - [ ] Global box center: X, Y, Z → cx, cy, cz
  - [ ] Box dimensions: X, Y, Z → dx, dy, dz
- [ ] type (vertical/horizontal/platform) 판단
- [ ] CSV에 한 줄 추가
- [ ] 원본 복구 후 다음 부재로

작업 후:
- [ ] CSV 파일 저장 (.csv 확장자)
- [ ] 파일 열어서 형식 확인

---

## 예상 소요 시간

| 부재 개수 | 예상 시간 |
|-----------|-----------|
| 5개 | 15-20분 |
| 10개 | 30-40분 |
| 20개 | 1-1.5시간 |

처음에는 시간이 좀 걸리지만, 익숙해지면 부재당 2-3분이면 됩니다.
