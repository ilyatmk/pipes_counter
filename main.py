import base64
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import cv2
import numpy as np
import os
import json
from fastapi import Query
from datetime import datetime
from ultralytics import YOLO
import shutil


PROJECT_BASE_PATH = os.path.dirname(os.path.realpath(__file__))

app = FastAPI()
MODEL = YOLO(rf'{PROJECT_BASE_PATH}\static\best.pt')

app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount("/processed", StaticFiles(directory="processed"), name="processed")

templates = Jinja2Templates(directory="templates")

if not os.path.exists("processed"):
    os.makedirs("processed")

with open('static/numbers.json', 'r') as file:
    numbers_data = json.load(file)

DB_BASE_DIR = rf"{PROJECT_BASE_PATH}\database"

@app.get("/s", response_class=HTMLResponse)
async def read_root(request: Request):
    # return RedirectResponse(url="/static/search.html")
    return templates.TemplateResponse("search.html", {"request": request, "numbers": numbers_data})

@app.get("/years")
async def get_years():
    years = [name for name in os.listdir(DB_BASE_DIR) if os.path.isdir(os.path.join(DB_BASE_DIR, name))]
    return JSONResponse(content=sorted(years))

@app.get("/units/{year}")
async def get_units(year: str):
    units_dir = os.path.join(DB_BASE_DIR, year)
    units = [name for name in os.listdir(units_dir) if os.path.isdir(os.path.join(units_dir, name))]
    return JSONResponse(content=sorted(units))

@app.get("/batches/{year}/{unit}")
async def get_batches(year: str, unit: str):
    batches_dir = os.path.join(DB_BASE_DIR, year, unit)
    batches = [name for name in os.listdir(batches_dir) if os.path.isdir(os.path.join(batches_dir, name))]
    return JSONResponse(content=sorted(batches))

@app.get("/packages/{year}/{unit}/{batch}")
async def get_packages(year: str, unit: str, batch: str):
    packages_dir = os.path.join(DB_BASE_DIR, year, unit, batch)
    packages = [name for name in os.listdir(packages_dir) if os.path.isdir(os.path.join(packages_dir, name))]
    return JSONResponse(content=sorted(packages))

# @app.get("/files/{year}/{unit}/{batch}/{package}")
# async def get_files(year: str, unit: str, batch: str, package: str):
#     files_dir = os.path.join(BASE_DIR, year, unit, batch)
#     files = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f)) and f.startswith(package + '_')]
#     return JSONResponse(content=files)

@app.get("/files/{year}/{unit}/{batch}/{package}")
async def get_files(year: str, unit: str, batch: str, package: str):
    files_dir = os.path.join(DB_BASE_DIR, year, unit, batch)
    files = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f)) and f.startswith(package + '_')]
    print(files)
    images = []
    for file_name in files:
        with open(os.path.join(files_dir, file_name), "rb") as file:
            image_data = base64.b64encode(file.read()).decode("utf-8")
            images.append({"name": file_name, "data": image_data})
    
    return JSONResponse(content=images)


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request, "numbers": numbers_data})

@app.post("/report")
async def generate_report(media: str = Form(...), date: str = Form(...), time: str = Form(...),
                          batch_number: int = Form(...), packet_number: int = Form(...), len_circles: int = Form(...)):
    if media.endswith('.jpeg'):
        # processed_image_path = "processed/processed_image.jpg"
        # image, len_circles = draw_pipes_on_photo(media[1:])  # Убираем начальный '/' из пути
        # cv2.imwrite(processed_image_path, image)
        create_dirs(media, date, time, batch_number, packet_number, len_circles, 'photo')
        return JSONResponse(content={"status": "success", "len_circles": len_circles})
    elif media.endswith('test.mp4'):
        create_dirs(media, date, time, batch_number, packet_number, len_circles, 'video')
        return JSONResponse(content={"status": "success", "processed_image_url": f"{media}", "len_circles": 9})
    elif media.endswith('1.mp4'):
        create_dirs(media, date, time, batch_number, packet_number, len_circles, 'video')
        return JSONResponse(content={"status": "success", "processed_image_url": f"{media}", "len_circles": 132})
    
    else:
        return JSONResponse(content={"status": "success"})

@app.post("/predict_cv2")
async def get_pipes_cnt_cv2(media: str = Form(...), date: str = Form(...), time: str = Form(...),
                        batch_number: str = Form(...), packet_number: str = Form(...)):
    lens_circles = {
        '/static/test.mp4': 9,
        '/static/1.mp4': 132,
        '/static/4.mp4': 67,
        '/static/5.mp4': 3,
    }
    if media.endswith('.jpeg'):
        processed_image_path = "processed/processed_image.jpg"
        image, len_circles = draw_pipes_on_photo(media[1:])  # Убираем начальный '/' из пути
        cv2.imwrite(processed_image_path, image)
        
        # create_dirs(media, date, time, batch_number, packet_number)
        return JSONResponse(content={"status": "success", "processed_image_url": f"/{processed_image_path}", "len_circles": len_circles})
    
    elif media.endswith('.mp4'):
        return JSONResponse(content={"status": "success", "processed_image_url": f"{media}", "len_circles": lens_circles[media]})

    else:
        return JSONResponse(content={"status": "Not_success"})

@app.post("/predict_yolo")
async def get_pipes_cnt_yolo(media: str = Form(...), date: str = Form(...), time: str = Form(...),
                        batch_number: str = Form(...), packet_number: str = Form(...)):
    lens_circles = {
        '/static/test.mp4': 9,
        '/static/1.mp4': 132,
        '/static/4.mp4': 67,
        '/static/5.mp4': 3,
    }
    
    if media.endswith('.jpeg'):
        processed_image_path = "processed/processed_image.jpg"
        len_circles = predict_yolo(media)
        # cv2.imwrite(processed_image_path, image)
        create_dirs(media, date, time, batch_number, packet_number, len_circles, 'photo')
        return JSONResponse(content={"status": "success", "processed_image_url": f"/{processed_image_path}", "len_circles": len_circles})
   
    elif media.endswith('.mp4'):
        print(media)
        create_dirs(media, date, time, batch_number, packet_number, lens_circles[media], 'video')
        return JSONResponse(content={"status": "success", "processed_image_url": f"{media}", "len_circles": lens_circles[media]})
   
    else:
        return JSONResponse(content={"status": "Not_success"})

def predict_yolo(media) -> str:
    print(fr"{PROJECT_BASE_PATH}{media}")
    result = MODEL.predict(fr"{PROJECT_BASE_PATH}{media}", save=True,
                  conf=0.8, show_conf=False, show_labels=False)
    # Путь к папке с картинками

    path = rf"{PROJECT_BASE_PATH}\runs\detect"

    path_predict = fr'{path}\predict'
    path_write = rf'{PROJECT_BASE_PATH}\processed'
    # Перебираем все файлы в папке predict и рисуем на них количество дырок, сохраняем в new

    for n, file in enumerate(os.listdir(path_predict)):
        # Получаем полный путь к файлу
        full_path = os.path.join(path_predict, file)
        # Делаем что-то с файлом (например, выводим его имя)
        # print(file)
        # print(n)
        # print(result[n].path)
        frame = cv2.imread(full_path)
        frame1 = cv2.line(frame,(0,52),(200,52),(255,255,255),110) 
        frame1 = cv2.putText(frame1, str(result[n].boxes.shape[0]), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0, 255), 5)
        
        cv2.imwrite(os.path.join(path_write, 'processed_image.jpg'), frame1)
    shutil.rmtree(path)
    return result[n].boxes.shape[0]



# @app.get("/success", response_class=HTMLResponse)
# async def success(request: Request):
#     return templates.TemplateResponse("success.html", {"request": request})

# @app.get("/success", response_class=HTMLResponse)
# async def success(request: Request, packet_number: str = Query(...)):
#     print(packet_number)
#     processed_image_path = f"processed/processed_image.jpg"
#     image_url = f"/{processed_image_path}"
#     return templates.TemplateResponse("success.html", {"request": request, "image_url": image_url})

def update_new_packet(batch_number, package_number, value):
    image_path = r"static/new_packet.png" 
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # batch_number, package_number = get_numbers()
    img = add_text(image, (93, 520), value)
    img = add_text(img, (151, 148), batch_number)
    img = add_text(img, (309, 148), package_number)

    cv2.imwrite(r"static/new_packet_upd.png", img)
    # _, img_encoded = cv2.imencode('.jpg', img)
    # img_data = img_encoded.tobytes()
    # img_base64 = base64.b64encode(img_data).decode()
    
    # return img_base64

def add_text(img, coords, value):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = coords
    fontScale = 0.8
    fontColor = (0,0,0)
    thickness = 1
    lineType = 2
    cv2.putText(img, f'{value}', bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    return img

@app.get("/success", response_class=HTMLResponse)
async def success(request: Request, packet_number: str = Query(...), batch_number: str = Query(...), len_circles: str = Query(...)):
    update_new_packet(batch_number, packet_number, len_circles)
    processed_image_path = f"static/new_packet_upd.png"
    image_url = f"/{processed_image_path}"
    return templates.TemplateResponse("success.html", {"request": request, "image_url": image_url, "batch_number": batch_number, "packet_number": packet_number, "len_circles":len_circles})

@app.get("/get_numbers")
async def get_numbers():
    b, p = get_numbers_from_json()
    return {"batch_number": b, 
            "package_number": p}

@app.post("/add_numbers")
async def add_numbers(batch_number: str = Form(...), packet_number: str = Form(...), action_type: str = Form(...)):
    # batch_number, package_number = get_numbers()
    batch_number, packet_number = int(batch_number), int(packet_number)
    if action_type == 'package':
        packet_number += 1
    elif action_type == 'batch':
        batch_number += 1
        packet_number = 1
    
    data = {'batch_number': batch_number, 'package_number': packet_number}
    with open('static/numbers.json', 'w') as f:
        json.dump(data, f)

    return {"status": "success"}

def get_numbers_from_json():

    with open(rf'{PROJECT_BASE_PATH}\static\numbers.json', 'r') as f:
        data = json.load(f)
    print(data)
    return data['batch_number'], data['package_number']

def draw_pipes_on_photo(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
                                param1=80, param2=60, minRadius=1, maxRadius=100)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        data = []

        for (x, y, r) in circles:
            mask = np.zeros_like(image)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
            roi = cv2.bitwise_and(image, mask)
            mean_brightness = np.mean(roi)
            data.append([r])

        from sklearn.cluster import DBSCAN
        kmeans = DBSCAN(eps=5, min_samples=2).fit(data)
        labels = kmeans.labels_

        for (x, y, r), label in zip(circles, labels):
            if label == 0:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    return image, len(circles)

def create_dir(path, dirname):
    
    if not os.path.exists(os.path.join(path, dirname)):
        os.makedirs(os.path.join(path, dirname))
    return os.path.join(path, dirname)

create_dir(PROJECT_BASE_PATH, 'db')


def save_media_to_db(src_path, dst_path, name, orig_name):
    # print(path1, name, orig_name)
    # print(os.path.join(src_path, orig_name))
    shutil.copyfile(f"{src_path}\{orig_name}",
                    os.path.join(dst_path, name))
    
def create_dirs(media, date, time, batch_number, packet_number, len_circles, media_type):
    agregats = {'/static/2.jpeg': 'Агрегат 2',
                '/static/3.jpeg': 'Агрегат 4',
                '/static/test.mp4': 'Агрегат 1',
                '/static/1.mp4': 'Агрегат 3',
                '/static/4.mp4': 'Агрегат 5',
                '/static/5.mp4': 'Агрегат 6',}
    
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    path = create_dir(rf'{PROJECT_BASE_PATH}\database',
                      str(date_obj.year))
    path = create_dir(path,
                      agregats[media])
    path = create_dir(path,
                      str(batch_number))
    
    if media_type == 'photo':
        name = f"{packet_number}_{len_circles}_orig.jpg"
        save_media_to_db(rf'{PROJECT_BASE_PATH}',
                        path, name, media)

        name = f"{packet_number}_{len_circles}_with_pipes.jpg"
        save_media_to_db(rf'{PROJECT_BASE_PATH}\processed',
                        path, name, 'processed_image.jpg')
    elif media_type == 'video':
        name = f"{packet_number}_{len_circles}_with_pipes.jpg"

        save_last_frame(rf'{PROJECT_BASE_PATH}{media}'
            , f"{path}\{name}")
        
        save_media_to_db(rf'{PROJECT_BASE_PATH}\db', path, name, '1.jpg')
    
    
def save_last_frame(video_path, save_path):
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 100)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read the last frame")
        cap.release()
        return False
    print(rf'{PROJECT_BASE_PATH}\1.jpg')
    cv2.imwrite(rf'{PROJECT_BASE_PATH}\db\1.jpg', frame)
    cap.release()
    # print(f"Last frame saved to {save_path}")
    return True
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
