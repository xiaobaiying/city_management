/**
 * Created by LXT on 2018/5/21.
 */
$(function()
{
    SidebarTabHandler.Init();
    /*
    $("#preProcess").show();
    $("#calculate").hide();
    $("#showNewWords").hide();
    $("#LDA").hide();
    $("#topics").hide();
    $("#cutWatting").hide();

*/
    $('.wait-span').hide();


});
var SidebarTabHandler={
    Init:function(){
        $(".tabItemContainer>li").click(function(){
            $(".tabItemContainer>li>a").removeClass("tabItemCurrent");
            $(".tabBodyItem").removeClass("tabBodyCurrent");
            $(this).find("a").addClass("tabItemCurrent");
            $($(".tabBodyItem")[$(this).index()]).addClass("tabBodyCurrent");
        });
    }
};
function getFileName(obj) {
    var name = getNamebyPath(obj.value);
    if (name.substring(name.lastIndexOf('.') + 1)!="txt"){
        actionShow("上传的文件必须为txt格式",3000);
    	delFile(obj);//删除对应文件
    }

    if(name.replace(/[^\x00-\xff]/g,"xx").length > 100){//文件名长度（包括后缀名）不得超过100字节
    	actionShow("文件名总长度不得超过100字节",3000);
    	delFile(obj);//删除对应文件
    }else{
    	$(obj).parent().parent().next().children(".fileDisplay").html(name);//显示文件名
        //$('td').children().siblings(".fileDisplay").val(name)
        $(obj).parent().parent().next().children(".fileDisplay").val(name)
    }

}
function getNamebyPath(path) {
    var pos1 = path.lastIndexOf('/');
    var pos2 = path.lastIndexOf('\\');
    var pos = Math.max(pos1, pos2)
    if (pos < 0)
        return path;
    else
        return path.substring(pos + 1);
}

function analyse(){
    var selectedFile = $("#fileId").get(0).files[0];
    var fileName = $('td').children().siblings(".fileDisplay").val();
    $("#btnCutwords").attr("disabled","true");
    $("#cutWatting").show();
    $("#cutWatting").css("color","blue");



 var params = {
                "path": fileName,

            };
    ajaxPost('../usrClass/cutWordsApi/data_process/',params,
            function(data) {
         data = JSON.parse(data);

         $("#btnCutwords").removeAttr("disabled");
        if(data.result == "success"){
            $("#cutWatting").html("分词完成");
            $("#cutWatting").css("color","green");

        }

            }
        );
}
