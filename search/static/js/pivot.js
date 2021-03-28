$(document).ready(function() {
	$('.collection-item').on('click', function(event) {
    
	event.preventDefault();


	$('#results').hide();
	$('#loader').removeClass("inactive");
	$('#loader').addClass("active");
	var form = $(this).parent();
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
			// var f = 'http://alimbic.com/search/' + jQuery.param(data1);
			var f = 'http://192.168.4.36:8000/search/' + jQuery.param(data1);
      		history.pushState(data1, null, f);
		},

		error : function(xhr, errmsf, err) {
            console.log(xhr);
            console.log(err);
			console.log('xhr.status + ": " + xhr.responseText');
		}
	});

      
    });
	
	
});

function post_pivot_search(item) {

  var pivot_cid = $(item).parent().children('#pivot_cid');


};

