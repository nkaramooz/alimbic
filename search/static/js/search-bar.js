var chip = {
    tag: 'chip content',
    image: '', //optional
  };



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
          'N. Engl. J. Med.': null,
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
        },
        limit: Infinity,
        minLength: 1
      }
    })
});

// window.addEventListener('popstate', function(e) {
//   console.log("test 123");
//   console.log(JSON.stringify(event.state));
// });

function post_search_text() {
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
        unmatched_terms : $('#unmatched_terms').val(),
  }
  console.log(data1);

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
      var f = 'http://127.0.0.1:8000/search/' + jQuery.param(data1);
      // console.log(jQuery.param(data1));
      // console.log(data1['journals'][0]);
      // console.log(chipInstance.chipsData);
      // history.pushState(data, null, 'http://127.0.0.1:8000/');
      history.replaceState(data1, null, f);
      history.pushState(data1, null, f);
    },

    error : function(xhr, errmsf, err) {
      console.log('xhr.status + ": " + xhr.responseText');
    }
  });
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