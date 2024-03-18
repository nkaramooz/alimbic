$(document).ready(function() {

	$(".dropdown-trigger").dropdown();
	
	$("#getAcid").on("submit", function(event) {
		event.preventDefault();
		getAcid();
	});

	$("#getAdid").on("submit", function(event) {
		event.preventDefault();
		getAdid();
	});

	$("#getTerm").on("submit", function(event) {
		event.preventDefault();
		getTerm();
	});

	$("#labelRelForm").on("submit", function(event) {
		event.preventDefault();
		labelRel();
	});

	$("#getRel").on("submit", function(event) {
		event.preventDefault();
		getRel();
	});

	$("#createConcept").on("submit", function(event) {
		event.preventDefault();
		createConcept();
	});

	$("#deactivateAcidForm").on("submit", function(event) {
		event.preventDefault();
		deactivateConcept();
	});

	$("#addDescriptionForm").on("submit", function(event) {
		event.preventDefault();
		addDescription();
	});

	$("#addDescriptionForm").on("submit", function(event) {
		event.preventDefault();
		addDescription();
	});

	$("#deactivateDescriptionForm").on("submit", function(event) {
		event.preventDefault();
		deactivateDescription();
	});

	$("#updateRelationshipFrom").on("submit", function(event) {
		event.preventDefault();
		updateRelationship();
	});

	$("#setAcronymForm").on("submit", function(event) {
		event.preventDefault();
		setAcronym();
	});

	$("#changeConceptTypeForm").on("submit", function(event) {
		event.preventDefault();
		changeConceptType();
	});
});

//Calls server using parameters provided from each form submission
//Only used for calls retrieving read data from db
//Clear data is a boolean flag to trigger clearing of all previous
//results listed in prior calls.
function getData(params) {
	const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value; // From Django documentation
	$.ajax({
		url: params["url"],
		type: "POST",
		dataType: "json",
		data: params["payload"],
		headers: {'X-CSRFToken': csrftoken},
		mode: 'same-origin',

		success: function(data) {
			const clearResults = params.clearResults ?? true;
			if (clearResults) clearData();
			data["data"].forEach(function(item) {
				$(item["results-div"]).html(item["html"]);
			})
		},
		error: function(xhr, errmsf, err) {
			console.log("endpoint failed");
		}
	});
};

function getAcid() {
	const payload = {"acid": $("#acid-search").val(), "results-div": "#acid-results"};
	const params = {
		"url": "/get_acid/",
		"payload": payload,
		"clearData": true
	};
	getData(params);	
};

function getAdid() {
	const payload = {"adid": $("#adid-search").val(), "results-div": "#adid-results"};
	const params = {
		"url": "/get_adid/",
		"payload": payload,
		"clearData": true
	};
	getData(params);
};

function getTerm() {
	const payload = {
		"term": $("#term-search").val(),
		"results-div" : "#term-results"
	};
	const params = {
		"url": "/get_term/",
		"payload": payload,
		"clearData": true
	};
	getData(params);
};

function getRel() {
	const payload = {
		"acid": $("#acid-rel-lookup").val(),
		"results-div": {
			"parents" : "#acid-rel-parents",
			"children" : "#acid-rel-children"},
	};
	const params = {
		"url": "/get_rel/",
		"payload": payload,
		"clearData": true
	};
	getData(params)
};

function getParents(acid_val) {
	const payload = {acid: acid_val};
	const url = "/get_parents/";
	const div = "#acid-rel-parents";
	const params = {
		"url" : url,
		"results-div": div,
		"payload": payload,
		"clearData": true
	};
	getData(params);
};

function getChildren(acid_val) {
	const payload = {acid: acid_val};
	const url = "/get_children/";
	const div = "#acid-rel-children";
	const params = {
		"url" : url,
		"results-div": div,
		"payload": payload,
		"clearData": false
	};
	getData(params);
};

function createConcept() {
	const payload = {
		"term": $("#new-concept").val(),
		"results-div" : "#new-concept-results"
	};
	const params = {
		"url": "/create_concept/",
		"payload": payload,
		"clearData": true
	};
	getData(params);
};

function deactivateConcept() {
	const payload = {
		"acid": $("#deactivate-acid").val(),
		"results-div" : "#deactivate-acid-results"
	};
	const params = {
		"url": "/deactivate_concept/",
		"payload": payload,
		"clearData": true
	};
	getData(params);
};

function addDescription() {
	const payload = {
		"term": $("#new-description").val(),
		"acid": $("#new-description-acid").val(),
		"results-div" : "#new-description-results"
	};
	const params = {
		"url": "/new_description/",
		"payload": payload,
		"clearData": true
	};
	getData(params);
};


function deactivateDescription() {
	const payload = {
		"adid": $("#deactivate-adid").val(),
		"results-div" : "#deactivate-description-results"
	};
	const params = {
		"url": "/deactivate_description/",
		"payload": payload,
		"clearData": true
	}
	getData(params);
};


function updateRelationship() {
	const payload = {
		"child-acid": $("#update-rel-child-acid").val(),
		"parent-acid": $("#update-rel-parent-acid").val(),
		"update-rel": $('input[name="update-rel"]:checked').val(),
		"results-div": "#update-rel-results"
	};
	const params = {
		"url" : "/modify_parent/",
		"payload": payload,
		"clearData": true
	};
	getData(params);
};

function setAcronym() {
	const payload = {
		"adid": $("#set-acronym-adid").val(),
		"set-acronym": $('input[name="set-acronym"]:checked').val(),
		"results-div": "#set-acronym-results"
	};
	const params = {
		"url" : "/set_acronym/",
		"payload": payload,
		"clearData": true
	};
	getData(params);
};

function changeConceptType() {
	const payload = {
		"acid": $("#acid-change-concept-type").val(),
		"concept-type": $('input[name="set-concept-type"]:checked').val(),
		"state":  $('input[name="concept-type-state"]:checked').val(),
		"results-div": "#change-concept-type-results"
	};
	const params = {
		"url" : "/set_concept_type/",
		"payload": payload,
		"clearData": true
	};
	getData(params);
};


function labelRel() {
	const payload = {
		"condition_acid": $("#rel-condition-acid").val(),
		"treatment_acid": $("#rel-treatment-acid").val(),
		"rel": $('input[name="rel"]:checked').val(),
		"results-div": "#labelRel-results"
	};
	const params = {
		"url": "/post_label_rel/",
		"payload": payload,
		"clearData": true
	};
	getData(params);
};

//Clear search terms and results
function clearData() {
	$(".results").html("");
	$(".input").val("");
	$(".radio").prop('checked', false);
}