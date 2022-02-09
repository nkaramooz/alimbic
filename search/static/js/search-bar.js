var chip = {
    tag: 'chip content',
    image: '', //optional
  };

window.addEventListener('popstate', function(event) {
  document.getElementById('search_box').disabled = true;
  document.getElementById('start_year').disabled = true;
  document.getElementById('end_year').disabled = true;
  $('#results').hide();
  $('#loader').removeClass("inactive");
  $('#loader').addClass("active");

  if (event.state !== null) {
    $.ajax({
      url : "search/",
      type : "POST",
      contentType: 'application/json',
      data : JSON.stringify(event.state),


      success : function(json) {
        chips = event.state['journals'];
        $('.chip .close').click();
        var chipInstance = M.Chips.getInstance($(".chips-autocomplete"));
        for (let i=0; i < chips.length; i++) {
          chipInstance.addChip(chips[i]);
        };
        
        $('#start_year').val(event.state['start_year']);
        $('#end_year').val(event.state['end_year']);
        $('#query_type').val(event.state['query_type']);
        $('#query_annotation').val(event.state['query_annotation']);
        $('#expanded_query_acids').val(event.state['expanded_query_acids']);
        $('#unmatched_terms').val(event.state['unmatched_terms']);
        $('#pivot_cid').val(event.state['pivot_cid']);
        $('#pivot_term').val(event.state['pivot_term']);
        $('#search_box').val(event.state['query']);
        $('#search_box').focus();
        document.getElementById('search_box').disabled = false;
        document.getElementById('start_year').disabled = false;
        document.getElementById('end_year').disabled = false;
        document.getElementById('journals').disabled = false;
        $('#loader').removeClass("active");
        $('#loader').addClass("inactive");

        $("#results").html(json);
        $('#results').show();


    },

     error : function(xhr, errmsf, err) {
        console.log("error in pop");
        console.log(xhr);
        console.log("error");
        $('#loader').removeClass("active");
        $('#loader').addClass("inactive");
        $("#results").html("<div class=\"row s12\"><div class=\"col s12\" style=\"text-align: center\"> Something went wrong. Try another query </div> </div>")
        $('#results').show();
      }

    })
  }

  else {
    // window.location.replace('http://127.0.0.1:8000/');
    window.location.replace('http://alimbic.com/')

  }});



$(document).ready(function() {
    $('#expand_filters').click(function() {
      var icon = this.querySelector('.chevron');
 
      if (icon.className.includes('rotate-down')) {
        icon.className = 'tiny material-icons chevron rotate-up';
        $('#filters').slideUp(500);
      } else if (icon.className.includes('rotate-up')) {
        icon.className = 'tiny material-icons chevron rotate-down';
        $('#filters').slideDown(500);
      } else { 
       icon.classList.add('rotate-down');
       $('#filters').slideDown(500);
     }
    });
    $('#post_search_text').on('submit', function(event) {
      event.preventDefault();
      post_search_text();
    });
    $('#start_year').on('keypress', function(event) {
      if(event.code == 13) {
        event.preventDefault();
        post_search_text();
      }
    });
    $('#end_year').on('keypress', function(event) {
      if(event.code == 13) {
        event.preventDefault();
        post_search_text();
      }
    });
    
    $('.collapsible').collapsible();
    $('.chips').chips();
    $('.chips-autocomplete').chips({
      placeholder: 'Enter journals',
      autocompleteOptions: {
        data: {
          'Brain': null,
          'Heart Lung': null,
          'CMAJ': null,
          'Prog Cardiovasc Dis': null,
          'Obstet Gynecol': null,
          'Nurs Outlook': null,
          'JAMA Otolaryngol Head Neck Surg': null,
          'J Gerontol B Psychol Sci Soc Sci': null,
          'Crit. Care Med.': null,
          'Am. J. Surg.': null,
          'Dis Mon': null,
          'Clin Toxicol (Phila)': null,
          'Pediatrics': null,
          'Dig. Dis. Sci.': null,
          'Arch. Dis. Child. Fetal Neonatal Ed.': null,
          'J. Allergy Clin. Immunol.': null,
          'Am J Phys Med Rehabil': null,
          'AJR Am J Roentgenol': null,
          'Gastroenterology': null,
          'J Nurs Adm': null,
          'Ann. Thorac. Surg.': null,
          'Dig. Dis. Sci.': null,
          'J. Clin. Endocrinol. Metab.': null,
          'Chest': null,
          'JAMA Ophthalmol': null,
          'J. Nerv. Ment. Dis.': null,
          'J. Immunol.': null,
          'J Bone Joint Surg Am': null,
          'Ann. Intern. Med.': null,
          'Am. J. Ophthalmol.': null,
          'BMJ': null,
          'Am. J. Respir. Crit. Care Med.': null,
          'Anesthesiology': null,
          'Acad Med': null,
          'Br J Radiol': null,
          'Am J Psychiatry': null,
          'PLoS ONE': null,
          'J. Allergy Clin. Immunol.': null,
          'Endocrinology': null,
          'Radiol. Clin. North Am.': null,
          'JAMA Intern Med': null,
          'Am. J. Clin. Pathol.': null,
          'Am. J. Pathol.': null,
          'Orthop. Clin. North Am.': null,
          'JAMA Dermatol': null,
          'Anaesthesia': null,
          'Postgrad Med': null,
          'Am. Heart J.': null,
          'J. Am. Coll. Cardiol.': null,
          'Mayo Clin. Proc.': null,
          'J Trauma Acute Care Surg': null,
          'N Engl J Med': null,
          'Plast. Reconstr. Surg.': null,
          'Am. J. Clin. Nutr.': null,
          'Surg. Clin. North Am.': null,
          'Br J Surg': null,
          'South. Med. J.': null,
          'Medicine (Baltimore)': null,
          'J. Urol.': null,
          'Am J Nurs': null,
          'BJOG': null,
          'Arch. Dis. Child.': null,
          'JACC Heart Fail': null,
          'Nurs. Clin. North Am.': null,
          'J. Thorac. Cardiovasc. Surg.': null,
          'CA Cancer J Clin': null,
          'Heart Lung': null,
          'J. Urol.': null,
          'Nurs Res': null,
          'JAMA': null,
          'Diabetes': null,
          'Am. J. Obstet. Gynecol.': null,
          'South. Med. J.': null,
          'Rheumatology (Oxford)': null,
          'J. Am. Coll. Surg.': null,
          'Pediatr. Clin. North Am.': null,
          'J. Clin. Invest.': null,
          'Clin. Orthop. Relat. Res.': null,
          'J. Gerontol. A Biol. Sci. Med. Sci.': null,
          'J. Nerv. Ment. Dis.': null,
          'Ann. Surg.': null,
          'J. Clin. Invest.': null,
          'Orthop. Clin. North Am.': null,
          'Med. Clin. North Am.': null,
          'J Laryngol Otol': null,
          'Am. J. Ophthalmol.': null,
          'Hypertension': null,
          'J. Pediatr.': null,
          'Med Lett Drugs Ther': null,
          'Am. J. Med.': null,
          'Nurs Outlook': null,
          'Ann. Otol. Rhinol. Laryngol.': null,
          'J Acad Nutr Diet': null,
          'Acad Med': null,
          'N. Engl. J. Med.': null,
          'JACC Heart Fail': null,
          'J Fam Pract': null,
          'Br J Radiol': null,
          'Am. J. Cardiol.': null,
          'Transl Res': null,
          'Radiology': null,
          'JAMA Neurol': null,
          'Clin Pediatr (Phila)': null,
          'J. Neurosurg.': null,
          'Am. J. Med. Sci.': null,
          'Diabetes': null,
          'J Trauma Acute Care Surg': null,
          'Am. Heart J.': null,
          'Am J Public Health': null,
          'J. Oral Maxillofac. Surg.': null,
          'Ann. Thorac. Surg.': null,
          'Heart': null,
          'Neurology': null,
          'Med. Clin. North Am.': null,
          'BMJ Case Rep': null,
          'Blood': null,
          'Nurs. Clin. North Am.': null,
          'Gut': null,
          'Phys Ther': null,
          'Rheumatology (Oxford)': null,
          'Am J Psychiatry': null,
          'BJOG': null,
          'Clin. Pharmacol. Ther.': null,
          'CA Cancer J Clin': null,
          'Am. J. Clin. Pathol.': null,
          'Bone Joint J': null,
          'Am. J. Med. Sci.': null,
          'JAMA Neurol': null,
          'J Gerontol B Psychol Sci Soc Sci': null,
          'Ann. Surg.': null,
          'Urol. Clin. North Am.': null,
          'J Fam Pract': null,
          'Mayo Clin. Proc.': null,
          'J. Pediatr.': null,
          'BMJ': null,
          'J. Clin. Pathol.': null,
          'Urol. Clin. North Am.': null,
          'Arch Phys Med Rehabil': null,
          'JAMA Dermatol': null,
          'Surgery': null,
          'JAMA': null,
          'J Bone Joint Surg Am': null,
          'Curr Probl Surg': null,
          'Br J Surg': null,
          'Gastroenterology': null,
          'Lancet': null,
          'JAMA Surg': null,
          'Heart': null,
          'JAMA Psychiatry': null,
          'J Nurs Adm': null,
          'Postgrad Med': null,
          'Transl Res': null,
          'JAMA Pediatr': null,
          'Anesthesiology': null,
          'Clin. Pharmacol. Ther.': null,
          'Brain': null,
          'Curr Probl Surg': null,
          'Pediatrics': null,
          'J Bone Joint Surg Am': null,
          'Am. J. Cardiol.': null,
          'CMAJ': null,
          'Hypertension': null,
          'Arch. Pathol. Lab. Med.': null,
          'Arthritis Rheumatol': null,
          'Am J Nurs': null,
          'J Laryngol Otol': null,
          'JAMA Psychiatry': null,
          'Ann. Intern. Med.': null,
          'Chest': null,
          'Clin Toxicol (Phila)': null,
          'Bone Joint J': null,
          'Am Fam Physician': null,
          'Am Fam Physician': null,
          'J. Clin. Endocrinol. Metab.': null,
          'Obstet Gynecol': null,
          'Lancet': null,
          'Med Lett Drugs Ther': null,
          'Anesth. Analg.': null,
          'Clin Pediatr (Phila)': null,
          'Arch Environ Occup Health': null,
          'J. Am. Coll. Surg.': null,
          'J. Infect. Dis.': null,
          'J. Clin. Pathol.': null,
          'Crit. Care Med.': null,
          'Dis Mon': null,
          'Radiology': null,
          'Arch. Dis. Child.': null,
          'Anesth. Analg.': null,
          'Prog Cardiovasc Dis': null,
          'JAMA Surg': null,
          'JAMA Pediatr': null,
          'J. Immunol.': null,
          'Am. J. Clin. Nutr.': null,
          'Cancer': null,
          'Nurs Res': null,
          'Surgery': null,
          'Surg. Clin. North Am.': null,
          'Am. J. Trop. Med. Hyg.': null,
          'Am. J. Pathol.': null,
          'Circulation': null,
          'Am. J. Obstet. Gynecol.': null,
          'AJR Am J Roentgenol': null,
          'Endocrinology': null,
          'Arch Phys Med Rehabil': null,
          'J. Oral Maxillofac. Surg.': null,
          'J. Infect. Dis.': null,
          'Hosp Pract (1995)': null,
          'Am. J. Trop. Med. Hyg.': null,
          'Arch. Pathol. Lab. Med.': null,
          'Ann Emerg Med': null,
          'Am J Phys Med Rehabil': null,
          'Ann Emerg Med': null,
          'Anaesthesia': null,
          'J. Am. Coll. Cardiol.': null,
          'Plast. Reconstr. Surg.': null,
          'Phys Ther': null,
          'Circulation': null,
          'Clin. Orthop. Relat. Res.': null,
          'Am. J. Respir. Crit. Care Med.': null,
          'Public Health Rep': null,
          'Cancer': null,
          'JAMA Intern Med': null,
          'Medicine (Baltimore)': null,
          'Arch. Dis. Child. Fetal Neonatal Ed.': null,
          'Am J Public Health': null,
          'Cochrane Database Syst Rev': null,
          'J. Neurosurg.': null,
          'J. Gerontol. A Biol. Sci. Med. Sci.': null,
          'J. Thorac. Cardiovasc. Surg.': null,
          'Blood': null,
          'Am. J. Surg.': null,
          'Ann. Otol. Rhinol. Laryngol.': null,
          'Neurology': null,
          'Public Health Rep': null,
          'Gut': null,
          'Radiol. Clin. North Am.': null,
          'JAMA Ophthalmol': null,
          'Pediatr. Clin. North Am.': null,
          'Arthritis Rheumatol': null,
          'Am. J. Med.': null,
          'JAMA Otolaryngol Head Neck Surg': null,
          'J Am Coll Cardiol' : null,
        },
        limit: Infinity,
        minLength: 1
      }
    })
});


function post_search_text() {
  $(':focus').blur()
  document.getElementById('search_box').disabled = true;
  document.getElementById('start_year').disabled = true;
  document.getElementById('end_year').disabled = true;
  $('#results').hide();
  $('#loader').removeClass("inactive");
  $('#loader').addClass("active");
  var chipInstance = M.Chips.getInstance($(".chips-autocomplete"));


  var data1 = { query : $('#search_box').val(),
         start_year : $('#start_year').val(),
        end_year : $('#end_year').val(),
        journals : chipInstance.chipsData,
        query_type : "keyword",
        query_annotation : $('#query_annotation').val(),
        expanded_query_acids : $('#expanded_query_acids').val(),
        unmatched_terms : $('#unmatched_terms').val(),
        pivot_cid : null,
        pivot_term : null,
  }

  $.ajax({
    url : "search/",
    type : "POST",
    contentType: 'application/json',
    data : JSON.stringify(data1),

    success : function(json) {
      document.getElementById('search_box').disabled = false;
      document.getElementById('start_year').disabled = false;
      document.getElementById('end_year').disabled = false;
      document.getElementById('journals').disabled = false;
      $('#loader').removeClass("active");
      $('#loader').addClass("inactive");
      $("#results").html(json);
      $('#results').show();
      var f = 'http://alimbic.com/search/' + jQuery.param(data1);
      // var f = 'http://127.0.0.1:8000/search/' + jQuery.param(data1);
      history.pushState(data1, null, f);
      
    },

    error : function(xhr, errmsf, err) {
      console.log(xhr);
      console.log("error");
      console.log("error here");
      document.getElementById('search_box').disabled = false;
      document.getElementById('start_year').disabled = false;
      document.getElementById('end_year').disabled = false;
      document.getElementById('journals').disabled = false;
      $('#loader').removeClass("active");
      $('#loader').addClass("inactive");
      $("#results").html("<div class=\"row s12\"> <div class=\"col s12\" style=\"text-align: center\"> Something went wrong. Try another query </div> </div>")
      $('#results').show();
    }
  });
};

$(function() {
    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie != '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                if (cookie.substring(0, name.length + 1) == (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    var csrftoken = getCookie('csrftoken');


    function csrfSafeMethod(method) {
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }
    function sameOrigin(url) {
        var host = document.location.host; // host + port
        var protocol = document.location.protocol;
        var sr_origin = '//' + host;
        var origin = protocol + sr_origin;
        return (url == origin || url.slice(0, origin.length + 1) == origin + '/') ||
            (url == sr_origin || url.slice(0, sr_origin.length + 1) == sr_origin + '/') ||
            !(/^(\/\/|http:|https:).*/.test(url));
    }

    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && sameOrigin(settings.url)) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });

});