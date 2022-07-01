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
	
	var narrowed_query_a_cids = [];
	$('[name="narrowed_query_a_cids[]"]').each(function() {
		narrowed_query_a_cids.push($(this).val());
	}).get();

	var unmatched_list = [];
	$('[name="unmatched_list[]"]').each(function() {
		unmatched_list.push($(this).val());
	}).get();

	var pivot_complete_acid = []
	form.find('#pivot_complete_acid').each(function(){
		pivot_complete_acid.push($(this).val());
	}).get()


	var chipInstance = M.Chips.getInstance($(".chips"));

	var data1 = { query : $('#search_box').val(),
			start_year : $('#start_year').val(),
			end_year : $('#end_year').val(),
			journals : chipInstance.chipsData,
			query_type : "pivot",
			query_annotation : query_list,
			narrowed_query_a_cids : narrowed_query_a_cids,
			pivot_complete_acid : pivot_complete_acid,
			unmatched_list : unmatched_list,
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
			// var f = 'http://127.0.0.1:8000/search/' + jQuery.param(data1);
      history.pushState(data1, null, f);
		},

		error : function(xhr, errmsf, err) {
            console.log(xhr);
            console.log(err);
			console.log('xhr.status + ": " + xhr.responseText');
			 $('#loader').removeClass("active");
        $('#loader').addClass("inactive");
        $("#results").html("<div class=\"row s12\"> <div class=\"col s12\" style=\"text-align: center\"> Something went wrong. Try another query </div> </div>")
        $('#results').show();
		}
	});

      
    });
	
	
});

function post_pivot_search(item) {
  var pivot_cid = $(item).parent().children('#pivot_cid');
};


