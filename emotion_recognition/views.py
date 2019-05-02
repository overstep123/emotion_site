from django.shortcuts import render
from emotion_recognition.models import Img,Video
from emotion_recognition.recognition import image_demo,video_recognition
from django.views.generic import ListView,DeleteView

def uploadImg(req): # 图片上传函数
    if req.method == 'POST':
        img = Img(img_url=req.FILES.get('img'))
        img.save()
        url = "media/"+str(img.img_url)
        imgUrl1 = "/"+url
        i = str(img.img_url).index(".")
        imgUrl="media/"+str(img.img_url)[:i]+"_recognition"+str(img.img_url)[i:]
        imgUrl2="/"+imgUrl
        req.close()
        if image_demo.save_predict(url,imgUrl)==True:
            return render(req, 'imgUpload.html', {"url1": imgUrl1, "url2": imgUrl2})
        else:
            return render(req, 'imgUpload.html',{"url1":imgUrl1,"url2":"","error":"1"})
    return render(req, 'imgUpload.html',)


def showImg(req):
    imgs = Img.objects.all() # 从数据库中取出所有的图片路径
    context = {
        'imgs' : imgs
    }
    print(imgs[0].img_url.url)
    Img.objects.all().delete()
    return render(req, 'showImg.html', context)

def uploadVdo(req):
    if req.method == 'POST':
        vv = Video.objects.all()
        vv.delete()
        vdo = Video(video_url =req.FILES.get('vdo'))
        vdo.save()
        url = "media/"+str(vdo.video_url)
        vdoUrl="media/"+str(vdo.video_url)[:-4]+"_rec"+str(vdo.video_url)[-4:]
        vdoUrl1="/"+vdoUrl
        video_recognition.videoRecognition(url,vdoUrl)
        req.close()
        return render(req, 'vdoUpload.html', {"url": vdoUrl1})
    return render(req, "vdoUpload.html")


def testD(req):
    return render(req,"test.html")

def index(req):
    return render(req,"index.html")
