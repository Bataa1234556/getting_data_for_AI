import os
from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from .excel_utils import read_excel
from .data_transfer import transfer_data

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            file_path = os.path.join(os.getcwd(), file.name)  # Use current working directory
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            # Read data from the Excel file
            read_excel(file_path)

            # Transfer data to the destination model
            transfer_data()

            # Delete the file after processing
            os.remove(file_path)

            return HttpResponseRedirect('/success/')
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})

def success(request):
    return render(request, 'success.html')
