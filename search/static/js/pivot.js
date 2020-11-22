$(document).ready(function() {
	$('.pivot-form').on('submit', function(event) {
	event.preventDefault();


	$('#results').hide();
	$('#loader').removeClass("inactive");
	$('#loader').addClass("active");
	var form = $(this);

	var query_list = [];
	$('[name="query_annotation[]"]').each(function() {
		query_list.push($(this).val());
	}).get();

	

	var chipInstance = M.Chips.getInstance($(".chips"));

	var data1 = { query : $('#search_box').val(),
			start_year : $('#start_year').val(),
			end_year : $('#end_year').val(),
			journals : chipInstance.chipsData,
			query_type : "pivot",
			query_annotation : query_list,
			unmatched_terms : $('#unmatched_terms').val(),
			pivot_cid : form.find('#pivot_cid').val(),
			pivot_term : form.find('#pivot_term').val(),
  		}

	$.ajax({
		url : "search/",
		type : "POST",
		contentType: 'application/json',
		data : JSON.stringify(data1),

		success : function(json) {
			$('#loader').removeClass("active");
			$('#loader').addClass("inactive");
			$('#search_box').val($('#search_box').val() + ' ' + form.find('#pivot_term').val());
			$("#results").html(json);
			$('#results').show();
			var f = 'http://alimbic.com/search/' + jQuery.param(data1);

			history.replaceState(data1, null, f);
      		history.pushState(data1, null, f);
		},

		error : function(xhr, errmsf, err) {
			console.log('xhr.status + ": " + xhr.responseText');
		}
	});

      

      

      // console.log($(this).children('#pivot_cid').first().val());
      // post_pivot_search(this);
    });
	// $('.expand-btn').on('click', function() {


	// 	console.log("clicked");
	// 	var prev = $(this).parent().parent().find(".preview-ellipsis")

	// 	if (prev.length > 0) {
	// 		prev.removeClass('preview-ellipsis').addClass('preview-full');
	// 	} else {
	// 		$(this).parent().parent().find('.preview-full').removeClass('preview-full').addClass('preview-ellipsis');
	// 	}

	// 	$(this).parent().parent().find(".expanded").toggle();

	// 	if ($(this).text() == 'Expand') {
	// 		$(this).html('<button id="expando" class="btn btn-light">Collapse</button>');
	// 	} else {
	// 		$(this).html('<button id="expando" class="btn btn-light">Expand</button>');
	// 	}
	// });
	
});

function post_pivot_search(item) {

  var pivot_cid = $(item).parent().children('#pivot_cid');


};

$(function() {


    // This function gets cookie with a given name
    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie != '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) == (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    var csrftoken = getCookie('csrftoken');

    /*
    The functions below will create a header with csrftoken
    */

    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }
    function sameOrigin(url) {
        // test that a given url is a same-origin URL
        // url could be relative or scheme relative or absolute
        var host = document.location.host; // host + port
        var protocol = document.location.protocol;
        var sr_origin = '//' + host;
        var origin = protocol + sr_origin;
        // Allow absolute or scheme relative URLs to same origin
        return (url == origin || url.slice(0, origin.length + 1) == origin + '/') ||
            (url == sr_origin || url.slice(0, sr_origin.length + 1) == sr_origin + '/') ||
            // or any other URL that isn't scheme relative or absolute i.e relative.
            !(/^(\/\/|http:|https:).*/.test(url));
    }

    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && sameOrigin(settings.url)) {
                // Send the token to same-origin, relative URLs only.
                // Send the token only if the method warrants CSRF protection
                // Using the CSRFToken value acquired earlier
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });

});
