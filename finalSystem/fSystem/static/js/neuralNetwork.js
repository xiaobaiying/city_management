
function train_st() {
    var keep_prob=$("#keep_prob_st").val();
    var hidden_nerual_size=$("#hidden_nerual_size_st").val();
    var learning_rate=$("#learning_rate_st").val();
    var num_epoch=$("#num_epoch_st").val();
    var modelName=$("#modelName_st").val();
    $("#btnTrain_st").attr("disabled","true");
    var params = {
     "keep_prob": keep_prob,
     "hidden_nerual_size": hidden_nerual_size,
     "learning_rate": learning_rate,
     "num_epoch": num_epoch,
     "modelName": modelName
    };
    ajaxPost('../usrClass/neuralNetwork/train_st/',params,
            function(data) {
         data = JSON.parse(data);

         $("#btnTrain_st").removeAttr("disabled");
        if(data.result == "success"){
            var s="训练完成！"+"<br/>"+"耗时(小时)："+data.time+"<br/>"+"训练结果:<br/>"+
                "max_f1:"+data.max_f1+"<br/>"+"precision:"+data.precision+"<br/>"+"recall:"+data.recall+"<br/>";
            $("#train_st_Result").html(s);
            $("#train_st_Result").css("color","green");

        }

            }
        );

}
function test_st() {
 var selectedtestFile = $("#test_st_fileId").get(0).files[0];
    var test_st_fileName = $('td').children().siblings(".test_st_fileDisplay").val();
    var selectedmodelFile = $("#test_model_st_fileId").get(0).files[0];
    var model_st_fileName = $('td').children().siblings(".test_model_st_fileDisplay").val();
    var outputFile=$("#test_st_result_fileName").val()
    $("#btnTest_st").attr("disabled","true");
    $("#test_st_Watting").show();
    $("#test_st_Watting").css("color","blue");
    var params = {
     "testFile": test_st_fileName,
     "modelName": model_st_fileName,
     "outputFile": outputFile

    };
    ajaxPost('../usrClass/neuralNetwork/test_st/',params,
            function(data) {
         data = JSON.parse(data);

         $("#btnTrain_st").removeAttr("disabled");
        if(data.result == "success"){
            var s="测试完成！ 耗时(小时)："+data.time;
            $("#test_st_Watting").html(s);
            $("#test_st_Watting").css("color","green");

        }

            }
        );

}
function train_tc() {
    var keep_prob=$("#keep_prob_tc").val();
    var hidden_nerual_size=$("#hidden_nerual_size_tc").val();
    var learning_rate=$("#learning_rate_tc").val();
    var num_epoch=$("#num_epoch_tc").val();
    var modelName=$("#modelName_tc").val();
    $("#btnTrain_tc").attr("disabled","true");
    var params = {
     "keep_prob": keep_prob,
     "hidden_nerual_size": hidden_nerual_size,
     "learning_rate": learning_rate,
     "num_epoch": num_epoch,
     "modelName": modelName
    };
    ajaxPost('../usrClass/neuralNetwork/train_tc/',params,
            function(data) {
         data = JSON.parse(data);

         $("#btnTrain_tc").removeAttr("disabled");
        if(data.result == "success"){
            var s="训练完成！"+"<br/>"+"耗时(小时)："+data.time+"<br/>"+"训练结果:<br/>"+
                "max_f1:"+data.max_f1+"<br/>"+"precision:"+data.precision+"<br/>"+"recall:"+data.recall+"<br/>";
            $("#train_tc_Result").html(s);
            $("#train_tc_Result").css("color","green");

        }

            }
        );

}
function test_tc() {
 var selectedtestFile = $("#test_tc_fileId").get(0).files[0];
    var test_tc_fileName = $('td').children().siblings(".test_tc_fileDisplay").val();
    var selectedmodelFile = $("#test_model_tc_fileId").get(0).files[0];
    var model_tc_fileName = $('td').children().siblings(".test_model_tc_fileDisplay").val();
    var outputFile=$("#test_tc_result_fileName").val()
    $("#btnTest_tc").attr("disabled","true");
    $("#test_tc_Watting").show();
    $("#test_tc_Watting").css("color","blue");
    var params = {
     "testFile": test_tc_fileName,
     "modelName": model_tc_fileName,
     "outputFile": outputFile

    };
    ajaxPost('../usrClass/neuralNetwork/test_tc/',params,
            function(data) {
         data = JSON.parse(data);

         $("#btnTrain_tc").removeAttr("disabled");
        if(data.result == "success"){
            var s="测试完成！ 耗时(小时)："+data.time;
            $("#test_tc_Watting").html(s);
            $("#test_tc_Watting").css("color","green");

        }

            }
        );

}
function getTopics(){
    $("#btnGenerate").attr("disabled","true")
    var topicDataPath='D:/Work/finalSystem/data/topicGenerate/'
    var params = {
     "file":topicDataPath+"inputData.txt",
     "outFile": topicDataPath+'sentenceAndTopics.txt'
    };
    ajaxPost('../usrClass/topic_generate/topicGenerate/',params,
            function(data) {
         data = JSON.parse(data);

         $("#btnGenerate").removeAttr("disabled");
        if(data.result == "success"){

            var s= "<tr><td>原句</td><td>主题</td></tr>"
            for(var i=0;i<data.info.length;i++){
                s+="<tr><td>"+data.info[i][0]+"</td><td>"+data.info[i][1]+"</td></tr>"
            }
             $("#topicsWait").hide();
             $("#topicsWait").after(s)

        }

            }
        );

}