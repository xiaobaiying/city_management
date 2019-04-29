function RemoteSupervision(){

    var modelfileName = $("#rsmodelfileName").val();
    var selectedtagFile = $("#rstagfileId").get(0).files[0];
    var tagfileName = $("#tagfileName").val();
    var taggedfileName=$('#rstaggedfileName').val();
    $("#btnRemoteSup").attr("disabled","true");
    $("#rsResult").show();
    $("#rsResult").css("color","blue");

 var params = {
     "modelfileName": modelfileName,
     "tagfileName": tagfileName,
     "taggedfileName": taggedfileName

            };
    ajaxPost('../usrClass/remoteSupervision/remoteSupervision/',params,
            function(data) {
         data = JSON.parse(data);

         $("#btnRemoteSup").removeAttr("disabled");
        if(data.result == "success"){
            $("#rsResult").html("自动标注完成，共有"+data.taggedWordNum+'个标注词组。');
            $("#rsResult").css("color","green");

        }

            }
        );
}