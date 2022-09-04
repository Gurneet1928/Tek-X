from django.shortcuts import render, HttpResponse
from videou.models import Video
from django.db.models import Q
from videou.forms import Video_form

# Create your views here.


def index(request):
    if 'search' in request.GET:
        q = request.GET['search']
        multiple_q = Q(Q(summary__icontains=q) | Q(summary__startswith=q))
        video = Video.objects.filter(multiple_q)
    else:
        video = Video.objects.all()
    context = {
        "video": video
    }
    return render(request, "index.html", context)


def upload(request):
    all_video = Video.objects.all()
    if request.method == "POST":
        form = Video_form(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponse("<h1> Uploaded successfully </h1>")
    else:
        form = Video_form()
    return render(request, 'upload.html', {"form": form, "all": all_video})
