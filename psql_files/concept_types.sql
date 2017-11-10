set schema 'annotation';

drop table if exists concept_types;

create table concept_types as (
	select
		conceptid
		,concept_type
	from (
		select 
			supertypeid as conceptid, 
			case 
				when subtypeid = '373873005' then 'treatment' -- pharmaceutical / biologic product
				when subtypeid = '417176000' then 'treatment' -- growth substance
				when subtypeid = '17948008' then 'treatment' -- hematopoietic factor
				when subtypeid = '106205006' then 'treatment' -- hemostasis related substance
				when subtypeid = '373244000' then 'treatment' -- immunologic substance ? cause
				when subtypeid = '85860001' then 'treatment' -- nervous system hormone-like substance
				when subtypeid = '35069000' then 'treatment' -- neurotransmitter
				when subtypeid = '226355009' then 'treatment' -- nutrients
				when subtypeid = '226355009' then 'treatment' -- nutrients
				when subtypeid = '362958002' then 'treatment' -- procedure by site
				when subtypeid = '180045004' then 'treatment' -- Amputation stump procedure
				when subtypeid = '127599007' then 'treatment' -- Application of hip spica cast
				when subtypeid = '408816000' then 'treatment' --  Artificial rupture of membranes
				when subtypeid = '238164007' then 'treatment' --  Body wall and cavity procedures
				when subtypeid = '363006003' then 'treatment' -- Cauterization by anatomic site
				when subtypeid = '363066000' then 'treatment' -- Destructive procedure by anatomic site
				when subtypeid = '363072000' then 'treatment' --  Diagnostic procedure by site
				when subtypeid = '363076002' then 'treatment' --  Diathermy procedure by body site
				when subtypeid = '447854000' then 'treatment' --  Dilation of enterostomy stoma
				when subtypeid = '75503001' then 'treatment' -- Excision of cyst of Gartner's duct
				when subtypeid = '31075001' then 'treatment' -- Excision of cyst of Müllerian duct in male 
				when subtypeid = '66136000' then 'treatment' -- Excision of Müllerian duct
				when subtypeid = '30700006' then 'treatment' -- Excision of transplanted tissue or organ
				when subtypeid = '8253009' then 'treatment' -- Excision of Wolffian duct
				when subtypeid = '66201006' then 'treatment' -- Extraction of fetus
				when subtypeid = '266783009' then 'treatment' -- Face to pubes conversion
				when subtypeid = '182656008' then 'treatment' -- General body warming therapy
				when subtypeid = '26667003' then 'treatment' -- Incision and packing of wound
				when subtypeid = '285837007' then 'treatment' -- Injection into body site
				when subtypeid = '76790002' then 'treatment' --  Insertion of tissue expander
				when subtypeid = '363186003' then 'treatment' --  Introduction of substance by body site 
				when subtypeid = '182655007' then 'treatment' --  Local heating - infrared irradiation
				when subtypeid = '363195006' then 'treatment' -- Manipulation procedure by body site
				when subtypeid = '177203002' then 'treatment' -- Manual removal of products of conception from delivered uterus
				when subtypeid = '9803001' then 'treatment' -- Medical procedure on body region
				when subtypeid = '229319000' then 'treatment' --  Mobilizing of body part 
				when subtypeid = '179986009' then 'treatment' -- Multisystem procedure 
				when subtypeid = '91097001' then 'treatment' -- Neuromuscular procedure
				when subtypeid = '448779009' then 'treatment' -- Occlusion of systemic to pulmonary artery shunt using transluminal embolic device 
				when subtypeid = '233233001' then 'treatment' -- Operation on systemic to pulmonary artery shunt 
				when subtypeid = '236994008' then 'treatment' -- Placental delivery procedure
				when subtypeid = '129152004' then 'treatment' --  Procedure on back 
				when subtypeid = '118664000' then 'treatment' -- Procedure on body system
				when subtypeid = '118949002' then 'treatment' -- Procedure on extremity 
				when subtypeid = '118754003' then 'treatment' -- Procedure on gland 
				when subtypeid = '118950002' then 'treatment' -- Procedure on head AND/OR neck
				when subtypeid = '118717007' then 'treatment' -- Procedure on organ 
				when subtypeid = '699465002' then 'treatment' -- Procedure on region of shoulder girdle
				when subtypeid = '118738001' then 'treatment' -- Procedure on soft tissue 
				when subtypeid = '118694006' then 'treatment' -- Procedure on trunk 
				when subtypeid = '78699000' then 'treatment' -- Radical amputation 
				when subtypeid = '429685005' then 'treatment' --  Radiotherapy by body site
				when subtypeid = '405450003' then 'treatment' -- Revision of hindquarter amputation
				when subtypeid = '307991006' then 'treatment' -- Revision of mastectomy scar
				when subtypeid = '363312001' then 'treatment' -- Stimulation procedure by body site
				when subtypeid = '363320004' then 'treatment' -- Surgical repair procedure by body site 
				when subtypeid = '442460002' then 'treatment' -- procedure on wound
				when subtypeid = '386637004' then 'treatment' -- obstetric procedure
				when subtypeid = '243120004' then 'treatment' -- Regimes and therapies 
				when subtypeid = '409002' then 'treatment' -- Food allergy diet 

				when subtypeid = '386811000' then 'diagnostic' -- Fetal procedure DIAGNOSTIC
				when subtypeid = '243773009' then 'diagnostic' -- Fetal blood sampling
				when subtypeid = '371571005' then 'diagnostic' -- Imaging by body site DIAGNOSTIC
				when subtypeid = '363244004' then 'diagnostic' -- Nuclear medicine study by site
				when subtypeid = '5880005' then 'diagnostic'  --  Physical examination procedure Diagnostic
				when subtypeid = '302381002' then 'diagnostic' -- Placental biopsy  Diagnostic
				when subtypeid = '285579008' then 'diagnostic' --  Taking swab from body site

				when subtypeid = '108252007' then 'diagnostic' -- laboratory procedure
				when subtypeid = '362961001' then 'diagnostic' -- procedure by intent

				when subtypeid = '410607006' then 'cause' -- organism
				when subtypeid = '88878007' then 'cause' -- protein
				when subtypeid = '106197002' then 'cause'
				when subtypeid = '106192008' then 'cause'
				when subtypeid = '7120007' then 'cause'

				when subtypeid = '404684003' then 'symptom' -- clinical finding

				when subtypeid = '123037004' then 'anatomy' -- body structure
				
				when subtypeid = '363787002' then 'observable' -- observable entity

				when subtypeid = '362981000' then 'qualifier' -- qualifier value

	   			when subtypeid = '64572001' then 'condition' -- disease
	   		end as concept_type

		from snomed.curr_transitive_closure_f
	) tb
	where concept_type is not null
);

create index ct_conceptid on concept_types(conceptid);
create index ct_concept_type on concept_types(concept_type)