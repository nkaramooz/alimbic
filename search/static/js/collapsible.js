document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.collapsible');
    var instances = M.Collapsible.init(elems, {});
  });


$(document).ready(function() {
    $('.collapsible').collapsible();
    $(".collapsible-header").click(function() {

    	if ($(this).find('.material-icons').html() == 'arrow_drop_down') {
    		$(this).find('.material-icons').html('arrow_drop_up');

    	}
    	else {
    		$(this).find('.material-icons').html('arrow_drop_down');
    	}

    })
});