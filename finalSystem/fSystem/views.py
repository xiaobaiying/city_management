# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

from django.template import loader,Context
from django.http import HttpResponse


import json
import sys
from django_ajax.decorators import ajax
import os
from fSystem.usrClass.cutWordsApi import cutWordsApi
from fSystem.usrClass.LDA import LDA
from fSystem.usrClass.neuralNetwork import neuralNetwork
from fSystem.usrClass.remoteSupervision import remoteSupervision
from fSystem.usrClass.topic_generate import topic_generate

apiClasses = [
    cutWordsApi,
    LDA,
    neuralNetwork,
    remoteSupervision,
    topic_generate



]
def main(request):
    return render(request,template_name = "main.html")
@ajax()
def upload(request):
    handle_uploaded_file(request.POST['fileObject'], str(request.POST['fileName']))
    return HttpResponse("Successful")
def handle_uploaded_file(file, filename):
    if not os.path.exists('upload/'):
        os.mkdir('upload/')

    with open('upload/' + filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
def getApiClasses(clsName):
    for apiCls in apiClasses:
        if apiCls.__name__.split('.')[-1] == clsName:
            return apiCls
    return None



@ajax()
def remoteView(request, apiClsName, methodName):
    cls = getApiClasses(apiClsName)
    if not cls: raise Exception("Can't find remote api class [%s]" % apiClsName)
    if cls and hasattr(cls, str(methodName)):#判断类中是否含有methodName方法
        params =request.POST
        #新建类对象
        inst = cls()
        inst.request = request
        m = getattr(inst, methodName)#引用类中的methodName方法
        if len(params) >0:
            params = dict([[str(k), str(v)] for k, v in params.items()])#将参数放入字典

            rs = m(**params)#调用对应方法
        else:
            rs = m()

        return HttpResponse(json.dumps(rs),content_type='application/json')

    raise Exception("Can't find remote method[%s] with api class [%s]" % (methodName, apiClsName))

