$(document).ready(function() {
	$('.js-example-basic-multiple').select2()

	$('#indication').on('change', function() {
		var indication = document.getElementById("indication").value;

		$.ajax({
			url: '/ajax/vcTroughTarget/',
			data: {
				'indication' : indication
			},
			dataType: 'json',
			success: function(data) {
				var troughTarget = data.trough.toString();
				$("#targetTrough").val(troughTarget);
			},
			error: function(data) {
				window.alert('failure')
			}
		})
	})
	
});

