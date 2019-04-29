/**
 * Created by LXT on 2018/6/7.
 */
function LDA(){
    $("#trLDAResult").show();
    $("#LDAResult").css("color","blue");
    $("#btnLDA").attr("disabled","true");
    $("#btncalCoWords").attr("disabled","true");

    $("#topicNum").attr("disabled","true");
    var topicNum = $("#topicNum").val();
    var fileName =$("#LDAfileName").val();
              ajaxPost('../usrClass/LDA/post_lda/', { "topic_num":topicNum,"fileName":fileName},
        function (LDAResult) {
            LDAResult= JSON.parse(LDAResult);

            if (LDAResult.result == "success") {
                $("#LDAResult").html(LDAResult.LDA.replace(/\n/g,"<br />"));
                $("#topicNum").val(topicNum);
                $("#LDAResult").css("color","black");
                $("#btnLDA").removeAttr("disabled");
                $("#btncalCoWords").removeAttr("disabled");

            } else {
                 $("#LDAResult").html(LDAResult);
                  $("#btnLDA").removeAttr("disabled");
                  $("#topicNum").removeAttr("disabled");
            }

        });
}
function SepCases() {
    var fileName =$("#LDAfileName").val();
    var wordsCasesFile=$("#wordsCasesFile").val();
    var wordsNotCasesFile=$("#wordsNotCasesFile").val();
    var wordList=$("#keyWordList").val();
    $("#sepCasesWatting").css("color","blue");



    ajaxPost('../usrClass/LDA/SepCases/', { "fileName":fileName,"wordsCasesFile":wordsCasesFile,
        "wordsNotCasesFile":wordsNotCasesFile,"wordList":wordList},
        function (result) {
            result= JSON.parse(result);

            if (result.result == "success") {
               $("#sepCasesWatting").html("分离完成");
               $("#sepCasesWatting").css("color","green");

            }

        });
}
function calCoWords() {
     $("#coWordsResult").css("color","blue");
    ajaxPost('../usrClass/LDA/calCoWords/', { },
        function (result) {
            result= JSON.parse(result);

            if (result.result == "success") {
                $("#coWordsResult").css("color","black");
                $("#coWordsResult").html(result.cowords);


            }

        });
}
function LDAkmeans() {
    $("#kmeansResult").css("color","blue");
    ajaxPost('../usrClass/LDA/KMeans/', {"fileName":"recentTopics.txt","n":15 },
        function (result) {
            result= JSON.parse(result);

            if (result.result.length>0) {
                $("#kmeansResult").css("color","black");
                $("#kmeansResult").html(result.result);


            }

        });
}
function showkg() {
    
}