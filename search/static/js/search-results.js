$(document).ready(function() {

	$('.expand-btn').on('click', function() {

		var prev = $(this).parent().parent().find(".preview-ellipsis")

		if (prev.length > 0) {
			prev.removeClass('preview-ellipsis').addClass('preview-full');
		} else {
			$(this).parent().parent().find('.preview-full').removeClass('preview-full').addClass('preview-ellipsis');
		}

		$(this).parent().parent().find(".expanded").toggle();

		if ($(this).text() == 'expand') {
			$(this).html('<button id="expando">collapse</button>');
		} else {
			$(this).html('<button id="expando">expand</button>');
		}
	});
	
});


