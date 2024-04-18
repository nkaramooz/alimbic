$(document).ready(function() {
	$('.collection-item').on('click', function(event) {
    
	event.preventDefault();

	$('#results').hide();
	$('#loader').removeClass("inactive");
	$('#loader').addClass("active");
	var form = $(this).parent();
	
	var nested_query_acids = [];
	var a_cid_id_array = [];
	var pivot_history = [];

  $("[id^='pivot_history']").each(function(){
      pivot_history.push($(this).val());
  });

	$("[id^='nested_query_acids']").each(function(){
		if(jQuery.inArray(this.id, a_cid_id_array) == -1) {
			a_cid_id_array.push(this.id);
		}
	});

	$.each( a_cid_id_array, function(key, value){
		sub_array = [];

		$("[id="+value+"]").each(function() {
				sub_array.push($(this).val());
			}).get();
			nested_query_acids.push(sub_array);
	});

	var unmatched_terms_list = [];
	$('[name="unmatched_terms_list[]"]').each(function() {
		unmatched_terms_list.push($(this).val());
	}).get();

	var pivot_complete_acid = []
	form.find('[name="pivot_complete_acid[]"').each(function(){
		pivot_complete_acid.push($(this).val());
	}).get()

	var chipInstance = M.Chips.getInstance($(".chips"));

	var data1 = { query : $('#search_box').val(),
			start_year : $('#start_year').val(),
			end_year : $('#end_year').val(),
			journals : chipInstance.chipsData,
			query_type : "pivot",
			nested_query_acids : nested_query_acids,
			pivot_complete_acid : pivot_complete_acid,
			pivot_history : pivot_history,
			unmatched_terms_list : unmatched_terms_list,
			pivot_concept : form.find('#pivot_concept').val(),
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


