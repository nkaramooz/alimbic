$(document).ready(function() {
	$('.expand-btn').on('click', function() {
		$(this).parent().parent().find(".expanded").toggle();
	});
});