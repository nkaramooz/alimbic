$(document).ready(function() {

	$('.expand_btn').on('click', function() {
		var prev = $(this).parent().parent().find(".preview-ellipsis")
		if (prev.length > 0) {
			prev.removeClass('preview-ellipsis').addClass('preview-full');
		} else {
			$(this).parent().parent().find('.preview-full').removeClass('preview-full').addClass('preview-ellipsis');
		}

		$(this).parent().find(".expanded").toggle();
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

	$('#more_treatment_btn').on('click', function() {
		if ($('#treatments').find('.hidden-items').hasClass('hide')) { 
			$('#treatments').find('.hidden-items').removeClass('hide');
			$(this).text('less');
		}
		else {
			$('#treatments').find('.hidden-items').addClass('hide');
			$(this).text('more');
		}			
	});

	$('#more_conditions_btn').on('click', function() {
		if ($('#conditions').find('.hidden-items').hasClass('hide')) { 
			$('#conditions').find('.hidden-items').removeClass('hide');
			$(this).text('less');
		}
		else {
			$('#conditions').find('.hidden-items').addClass('hide');
			$(this).text('more');
		}			
	});
	
});


