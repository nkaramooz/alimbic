$(document).ready(function() {

	$('.expand_btn').on('click', function() {
		console.log("clicked");
		var prev = $(this).parent().parent().find(".preview-ellipsis")
		console.log(prev)
		if (prev.length > 0) {
			prev.removeClass('preview-ellipsis').addClass('preview-full');
		} else {
			$(this).parent().parent().find('.preview-full').removeClass('preview-full').addClass('preview-ellipsis');
		}

		$(this).parent().find(".expanded").toggle();
		console.log($(this).find('#expando').attr('class'));
		if ($(this).find('#expando').hasClass('exp_more')) {
			$(this).find("#expand_icon").html('expand_less');
			$(this).find('#expando').removeClass('exp_more');
			$(this).add('#expando').addClass('exp_less');
		} else {
			$(this).find("#expand_icon").html('expand_more');
			$(this).find('#expando').removeClass('exp_less');
			$(this).find('#expando').addClass('exp_more');
		}
	});
	
});


