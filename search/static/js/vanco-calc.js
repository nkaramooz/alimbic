$(document).ready(function() {

	$('#vanco-calc').on('click', function() {
		var age = parseFloat(document.getElementById("age").value);
		var height_ft = parseFloat(document.getElementById("height-ft").value);
		var height_in = parseFloat(document.getElementById("height-in").value);
		var creatinine = parseFloat(document.getElementById("creatinine").value);
		var actualWeight = parseFloat(document.getElementById("weight").value);
		var indication = document.getElementById("indication").value;
		var troughTarget = document.getElementById("trough").value;

		var comorbid = $("#comorbid").val();


		if(document.getElementById('male').checked) {
			is_female = false;
		}
		else if(document.getElementById('female').checked) {
			is_female = true;
		}

		if(document.getElementById('loading').checked) {

			doseType='loading';
		}
		else if (document.getElementById('maintenance').checked) {
			doseType=('maintenance').toString();
		}

		$.ajax({
			url: '/ajax/vcSubmit/',
			data: {
				'age' : age,
				'is_female' : is_female,
				'height_ft' : height_ft,
				'height_in' : height_in,
				'creatinine' : creatinine,
				'actualWeight' : actualWeight,
				'indication' : indication,
				'troughTarget' : troughTarget,
				'doseType' : doseType,
				'comorbid' : comorbid
			},
			dataType:'json',
			success: function(data) {
				var crcl_text = ("CrCl : ").concat(data.crcl.toString());
				$("#crcl-text").html(crcl_text);
			},
			error: function(data) {
				window.alert('failure')
			}
		})
		
	});

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
				$("#trough").val(troughTarget);
			},
			error: function(data) {
				window.alert('failure')
			}
		})
	})
	
});


function returnIdealBodyWeight(is_female, height) {
	
	if (!is_female) {
		return 50+(2.3*height-60);
	}
	else {
		return 45.5+2.3*(height-60);
	}
}

function returnAdjustedWeightForIdealBodyWeight(idealBodyWeight, actualWeight) {
	return idealBodyWeight + 0.4*(actualWeight - idealBodyWeight);
}

function returnDosingWeight(is_female, height, actualWeight) {
	var dosingWeight = undefined

	var idealBodyWeight = returnIdealBodyWeight(is_female, height);

	if (actualWeight > (1.2*idealBodyWeight)) {
		return returnAdjustedWeightForIdealBodyWeight(idealBodyWeight, actualWeight);
	}
	else if (actualWeight < idealBodyWeight) {
		return actualWeight
	}
	else {
		return actualWeight
	}
}

function returnCrCl(age, is_female, height, actualWeight, creatinine) {
	var dosingWeight = returnDosingWeight(is_female, height, actualWeight);

	if ((age > 65) && (creatinine < 1)) {
		creatinine = 1
	}

	if (!is_female) {
		return (((140-age)*dosingWeight)/(72*creatinine)).toFixed(2);
	}
	else {

		return (((140-age)*dosingWeight*0.85)/(72*creatinine)).toFixed(2);
	}
}